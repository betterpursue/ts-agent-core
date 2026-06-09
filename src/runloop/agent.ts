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

      // 逐个执行工具调用（错误隔离：一个工具失败不影响其他工具）
      if (response.toolCalls && response.toolCalls.length > 0) {
        for (const tc of response.toolCalls) {
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
}
