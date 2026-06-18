/**
 * DefaultAgent — Agent 运行时核心实现
 *
 * 设计原则：
 * - 完整的 Tool Calling 循环：LLM → Tool → LLM → ... → Final Answer
 * - 错误的工具级隔离：一个工具挂了不影响整个 Agent
 * - 通过 Hooks 暴露扩展点，后续系列不需要修改核心
 * - 同一次 run() 内的消息用本地数组管理，不与记忆 eviction 策略耦合
 * - run() 结束后将对话同步到 ShortTermMemory，供下一次 run() 加载上下文
 * - stream() 和 run() 共享同一套消息管理逻辑，区别仅在 LLM 调用方式
 */

import type { Agent, AgentConfig, AgentResult, AgentHooks, AgentStreamEvent } from '../core/agent.js';
import type { Message } from '../core/message.js';
import { systemMessage, userMessage, assistantMessage } from '../core/message.js';
import type { SessionCheckpoint } from '../core/session.js';
import type { MemoryInjector } from '../core/memory.js';
import { executeToolsParallel } from '../tools/parallel-executor.js';

export class DefaultAgent implements Agent {
  readonly config: AgentConfig;
  hooks?: AgentHooks;

  /** 每个 session 的上次 checkpoint 消息数（用于 interval 策略） */
  private lastCheckpointCounts: Map<string, number> = new Map();
  /** 是否已经在当前会话中注入过记忆（用于 before_first_call 策略） */
  private injectedThisSession = false;

  constructor(config: AgentConfig) {
    this.config = config;
  }

  async run(input: string | Message[]): Promise<AgentResult> {
    return this.executeLoop(input, false);
  }

  async stream(
    input: string | Message[],
    onEvent: (event: AgentStreamEvent) => void
  ): Promise<AgentResult> {
    return this.executeLoop(input, true, onEvent);
  }

  reset(): void {
    this.config.memory.shortTerm.clear();
    this.injectedThisSession = false;
  }

  /**
   * 统一的执行循环
   *
   * run() 和 stream() 共享同一个核心循环。
   * 区别只有两点：
   * 1. LLM 调用方式：complete（非流式）vs completeStream（流式）
   * 2. stream 模式下通过 onEvent 推送中间事件
   *
   * 这样设计保证了两个方法的行为一致性：
   * - 工具调用、记忆同步、checkpoint 保存等逻辑完全复用
   * - 不会出现"run 能用但 stream 有 bug"的情况
   */
  private async executeLoop(
    input: string | Message[],
    streaming: boolean,
    onEvent?: (event: AgentStreamEvent) => void
  ): Promise<AgentResult> {
    const maxIterations = this.config.maxIterations ?? 10;
    const messages: Message[] = [];
    const session = this.config.session;

    // 1. 从 Session 加载已有消息
    if (session) {
      const sessionMessages = session.session.getMessages();
      messages.push(...sessionMessages);
    }

    // 2. 加载记忆中的历史对话
    const history = this.config.memory.shortTerm.getMessages();
    messages.push(...history);

    // 3. 系统提示词插到最前面
    if (this.config.systemPrompt) {
      messages.unshift(systemMessage(this.config.systemPrompt));
    }

    // 4. 追加用户输入，同时写入 Session
    if (typeof input === 'string') {
      const userMsg = userMessage(input);
      messages.push(userMsg);
      session?.session.addMessage(userMsg);
    } else {
      messages.push(...input);
      for (const msg of input) {
        session?.session.addMessage(msg);
      }
    }

    let iterations = 0;
    let totalTokens = 0;
    let stoppedEarly = false;
    let lastResponse = '';

    while (iterations < maxIterations) {
      iterations++;

      const toolMetadatas = this.config.tools.listMetadata();

      // 【记忆注入】在 LLM 调用前，将长期记忆注入上下文
      const llmMessages = await this.prepareMessagesWithMemory(messages);

      // Hook: LLM 调用前（注入后的消息列表）
      if (this.hooks?.onBeforeLLMCall) {
        await this.hooks.onBeforeLLMCall(llmMessages);
      }

      let response;

      if (streaming && onEvent) {
        // 流式模式：逐块推送 LLM 输出
        response = await this.config.model.provider.completeStream(
          {
            messages: llmMessages,
            model: this.config.model.model,
            tools: toolMetadatas.length > 0 ? toolMetadatas : undefined,
            maxTokens: this.config.model.maxTokens,
            temperature: this.config.model.temperature,
          },
          (chunk) => {
            if (chunk.contentDelta) {
              onEvent({ contentDelta: chunk.contentDelta });
            }
          }
        );
      } else {
        // 非流式模式
        response = await this.config.model.provider.complete({
          messages: llmMessages,
          model: this.config.model.model,
          tools: toolMetadatas.length > 0 ? toolMetadatas : undefined,
          maxTokens: this.config.model.maxTokens,
          temperature: this.config.model.temperature,
        });
      }

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

      // 执行工具调用
      if (response.toolCalls && response.toolCalls.length > 0) {
        // 推送 toolCall 事件
        if (streaming && onEvent) {
          for (const tc of response.toolCalls) {
            onEvent({
              toolCall: {
                id: tc.id,
                name: tc.name,
                args: tc.args,
              },
            });
          }
        }

        if (this.config.parallelExecution) {
          await this.executeToolCallsParallel(response.toolCalls, messages, streaming, onEvent);
        } else {
          await this.executeToolCallsSequential(response.toolCalls, messages, streaming, onEvent);
        }
      }

      // 推送 iteration 事件
      if (streaming && onEvent) {
        onEvent({ iteration: iterations });
      }

      // Hook
      if (this.hooks?.onIterationComplete) {
        await this.hooks.onIterationComplete(iterations);
      }

      // Checkpoint
      if (session?.checkpoints && session.checkpointTrigger) {
        await this.maybeCheckpoint(session);
      }

      // 没有 tool_calls → 最终答案
      if (!response.toolCalls || response.toolCalls.length === 0) {
        break;
      }
    }

    if (iterations >= maxIterations) {
      stoppedEarly = true;
    }

    // 将本轮对话写入短期记忆
    const conversationMessages = messages.filter((m) => m.role !== 'system');
    for (const msg of conversationMessages) {
      this.config.memory.shortTerm.add(msg);
    }

    // 同步到 Session
    if (session) {
      let lastUserIdx = -1;
      for (let i = 0; i < messages.length; i++) {
        if (messages[i].role === 'user' && messages[i].content !== '') {
          lastUserIdx = i;
        }
      }
      const newSessionMessages = messages.slice(lastUserIdx + 1).filter(
        (m) => m.role !== 'system'
      );
      for (const msg of newSessionMessages) {
        session.session.addMessage(msg);
      }

      if (session.checkpoints && session.checkpointTrigger?.type === 'manual') {
        const ckpt = session.session.checkpoint();
        await session.checkpoints.saveCheckpoint(session.session.id, ckpt);
      }
    }

    const result: AgentResult = {
      messages,
      output: lastResponse,
      iterations,
      totalTokens,
      stoppedEarly,
    };

    // 推送 done 事件
    if (streaming && onEvent) {
      onEvent({ done: result });
    }

    return result;
  }

  /**
   * 准备带有长期记忆注入的消息列表
   *
   * 如果配置了记忆注入，在系统提示之后插入一段记忆上下文消息。
   * 不对原始 messages 做修改，每次都返回新数组。
   */
  private async prepareMessagesWithMemory(
    messages: Message[]
  ): Promise<Message[]> {
    const injection = this.config.memoryInjection;
    if (!injection) {
      return messages; // 未配置注入，直接返回原始列表
    }

    const timing = injection.config?.timing ?? 'before_every_call';
    if (timing === 'before_first_call' && this.injectedThisSession) {
      return messages; // 已经注入过一次，跳过
    }

    const context = await injection.injector.buildContext(
      messages,
      this.config.memory.longTerm,
      injection.config
    );

    if (!context) {
      return messages; // 没有相关记忆
    }

    this.injectedThisSession = true;

    // 找到系统提示之后的位置插入记忆上下文
    // 如果有多条 system 消息，插在最后一条 system 之后
    const lastSystemIdx = messages.reduce((last, m, i) =>
      m.role === 'system' ? i : last, -1
    );

    const result = [...messages];
    result.splice(lastSystemIdx + 1, 0, {
      role: 'system',
      content: context,
    });

    return result;
  }

  /**
   * 检查是否需要保存检查点
   */
  private async maybeCheckpoint(
    session: NonNullable<AgentConfig['session']>
  ): Promise<void> {
    const trigger = session.checkpointTrigger!;

    switch (trigger.type) {
      case 'interval': {
        const msgCount = session.session.getMessages().length;
        const key = session.session.id;
        const lastCount = this.lastCheckpointCounts.get(key) ?? 0;
        if (msgCount - lastCount >= trigger.messageCount) {
          const ckpt = session.session.checkpoint();
          await session.checkpoints!.saveCheckpoint(session.session.id, ckpt);
          this.lastCheckpointCounts.set(key, msgCount);
        }
        break;
      }
      case 'tool_call':
      case 'manual':
        break;
    }
  }

  /**
   * 串行执行工具调用
   */
  private async executeToolCallsSequential(
    toolCalls: NonNullable<import('../core/message.js').ToolCall[]>,
    messages: Message[],
    streaming?: boolean,
    onEvent?: (event: AgentStreamEvent) => void
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

      // 推送 toolResult 事件
      if (streaming && onEvent) {
        onEvent({
          toolResult: {
            toolCallId: tc.id,
            toolName: tc.name,
            content: resultContent,
            isError,
          },
        });
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
   */
  private async executeToolCallsParallel(
    toolCalls: NonNullable<import('../core/message.js').ToolCall[]>,
    messages: Message[],
    streaming?: boolean,
    onEvent?: (event: AgentStreamEvent) => void
  ): Promise<void> {
    const calls = toolCalls.map((tc) => ({
      id: tc.id,
      name: tc.name,
      args: tc.args,
    }));

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
        resultContent = 'Error: Unknown execution failure';
        isError = true;
      }

      if (this.hooks?.onAfterToolCall) {
        await this.hooks.onAfterToolCall(tc.name, resultContent);
      }

      // 推送 toolResult 事件
      if (streaming && onEvent) {
        onEvent({
          toolResult: {
            toolCallId: tc.id,
            toolName: tc.name,
            content: resultContent,
            isError,
          },
        });
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
