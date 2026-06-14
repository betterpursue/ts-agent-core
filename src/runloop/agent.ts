/**
 * DefaultAgent — Agent 运行时核心实现
 *
 * 设计原则：
 * - 完整的 Tool Calling 循环：LLM → Tool → LLM → ... → Final Answer
 * - 错误的工具级隔离：一个工具挂了不影响整个 Agent
 * - 通过 Hooks 暴露扩展点，后续系列不需要修改核心
 * - 同一次 run() 内的消息用本地数组管理，不与记忆 eviction 策略耦合
 * - run() 结束后将对话同步到 ShortTermMemory，供下一次 run() 加载上下文
 */

import type { Agent, AgentConfig, AgentResult, AgentHooks } from '../core/agent.js';
import type { Message } from '../core/message.js';
import { systemMessage, userMessage, assistantMessage } from '../core/message.js';
import { executeToolsParallel } from '../tools/parallel-executor.js';

export class DefaultAgent implements Agent {
  readonly config: AgentConfig;
  hooks?: AgentHooks;

  constructor(config: AgentConfig) {
    this.config = config;
  }

  async run(input: string | Message[]): Promise<AgentResult> {
    const maxIterations = this.config.maxIterations ?? 10;
    const messages: Message[] = [];

    // 1. 加载记忆中的历史对话作为上下文
    const history = this.config.memory.shortTerm.getMessages();
    messages.push(...history);

    // 2. 系统提示词插到最前面
    if (this.config.systemPrompt) {
      messages.unshift(systemMessage(this.config.systemPrompt));
    }

    // 3. 追加用户输入
    if (typeof input === 'string') {
      messages.push(userMessage(input));
    } else {
      messages.push(...input);
    }

    let iterations = 0;
    let totalTokens = 0;
    let stoppedEarly = false;
    let lastResponse = '';

    while (iterations < maxIterations) {
      iterations++;

      // 获取当前已注册的所有工具元数据（用于构建 function calling schema）
      const toolMetadatas = this.config.tools.listMetadata();

      // Hook: LLM 调用前
      if (this.hooks?.onBeforeLLMCall) {
        await this.hooks.onBeforeLLMCall(messages);
      }

      // 调用 LLM
      const response = await this.config.model.provider.complete({
        messages,
        model: this.config.model.model,
        tools: toolMetadatas.length > 0 ? toolMetadatas : undefined,
        maxTokens: this.config.model.maxTokens,
        temperature: this.config.model.temperature,
      });

      // Hook: LLM 调用后
      if (this.hooks?.onAfterLLMCall) {
        await this.hooks.onAfterLLMCall(response);
      }

      totalTokens += response.usage?.totalTokens ?? 0;
      lastResponse = response.content;

      // 将 LLM 的回复加入对话
      messages.push(
        assistantMessage(response.content, response.toolCalls)
      );

      // 执行工具调用：并行（如果配置了）或串行（默认）
      if (response.toolCalls && response.toolCalls.length > 0) {
        if (this.config.parallelExecution) {
          await this.executeToolCallsParallel(response.toolCalls, messages);
        } else {
          await this.executeToolCallsSequential(response.toolCalls, messages);
        }
      }

      // Hook: 每轮迭代完成（无论是否有工具调用，都会触发）
      if (this.hooks?.onIterationComplete) {
        await this.hooks.onIterationComplete(iterations);
      }

      // 没有 tool_calls → 最终答案，循环结束
      if (!response.toolCalls || response.toolCalls.length === 0) {
        break;
      }
    }

    if (iterations >= maxIterations) {
      stoppedEarly = true;
    }

    // 将本轮对话写入短期记忆（排除 system prompt，因为每次 run() 都会重新 prepend）
    const conversationMessages = messages.filter((m) => m.role !== 'system');
    for (const msg of conversationMessages) {
      this.config.memory.shortTerm.add(msg);
    }

    return {
      messages,
      output: lastResponse,
      iterations,
      totalTokens,
      stoppedEarly,
    };
  }

  reset(): void {
    this.config.memory.shortTerm.clear();
  }

  /**
   * 串行执行工具调用（默认模式）
   *
   * 逐个执行，每个工具的结果作为 tool result message 加入对话。
   * 保持兼容性：这个执行方式在第 6 篇文章中详细解释过。
   */
  private async executeToolCallsSequential(
    toolCalls: NonNullable<import('../core/message.js').ToolCall[]>,
    messages: Message[]
  ): Promise<void> {
    for (const tc of toolCalls) {
      if (this.hooks?.onBeforeToolCall) {
        await this.hooks.onBeforeToolCall(tc.name, tc.args);
      }

      const tool = this.config.tools.get(tc.name);
      let resultContent: string;
      let isError = false;

      if (!tool) {
        const available = this.config.tools
          .listMetadata()
          .map((t) => t.name)
          .join(', ');
        resultContent = `Error: Tool '${tc.name}' not found. Available tools: ${available}`;
        isError = true;
      } else {
        try {
          const result = await tool.execute(tc.args);
          resultContent = result.output;
          isError = !result.success;
        } catch (err) {
          resultContent = `Error executing tool '${tc.name}': ${
            err instanceof Error ? err.message : String(err)
          }`;
          isError = true;
        }
      }

      if (this.hooks?.onAfterToolCall) {
        await this.hooks.onAfterToolCall(tc.name, resultContent);
      }

      messages.push({
        role: 'tool',
        content: resultContent,
        toolCallId: tc.id,
        toolName: tc.name,
        isError,
      });
    }
  }

  /**
   * 并行执行工具调用
   *
   * 和串行版本的核心区别：
   * 1. 所有工具同时触发执行（受 Semaphore 和依赖分析控制）
   * 2. 工具结果可能会以不同的顺序返回（按执行完成时间）
   * 3. 但推入消息列表时，仍然保持原始 tool_calls 的顺序
   *    ——这是 LLM API 的要求：tool result 的顺序必须与 tool_calls 一致
   *
   * 注意：onBeforeToolCall hooks 在所有工具执行前统一触发，
   * onAfterToolCall hooks 在收集到所有结果后按顺序触发。
   * 这保证了 hook 的语义正确性：before 在 execute 之前，after 在 execute 之后。
   */
  private async executeToolCallsParallel(
    toolCalls: NonNullable<import('../core/message.js').ToolCall[]>,
    messages: Message[]
  ): Promise<void> {
    const calls = toolCalls.map((tc) => ({
      id: tc.id,
      name: tc.name,
      args: tc.args,
    }));

    // onBeforeToolCall: 所有工具执行前统一触发
    if (this.hooks?.onBeforeToolCall) {
      for (const tc of toolCalls) {
        await this.hooks.onBeforeToolCall(tc.name, tc.args);
      }
    }

    const result = await executeToolsParallel(
      this.config.tools,
      calls,
      this.config.parallelExecution
    );

    // 按原始 tool_calls 顺序排列结果
    // 保持消息顺序一致，是 LLM 能正确理解 tool result 的关键
    for (const tc of toolCalls) {
      const toolResult = result.results.get(tc.id);
      const failure = result.failures.find((f) => f.id === tc.id);

      let resultContent: string;
      let isError = false;

      if (toolResult) {
        resultContent = toolResult.output;
        isError = !toolResult.success;
      } else if (failure) {
        resultContent = `Error: ${failure.error}`;
        isError = true;
      } else {
        // 兜底：不应该发生
        resultContent = 'Error: Unknown execution failure';
        isError = true;
      }

      if (this.hooks?.onAfterToolCall) {
        await this.hooks.onAfterToolCall(tc.name, resultContent);
      }

      messages.push({
        role: 'tool',
        content: resultContent,
        toolCallId: tc.id,
        toolName: tc.name,
        isError,
      });
    }
  }
}
