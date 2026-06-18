/**
 * Agent 接口 — 整个运行时的核心抽象
 *
 * 设计原则：
 * - Agent 不直接依赖具体实现，通过接口组合
 * - Run Loop 与 Agent 配置解耦，后续可以替换不同的循环策略
 * - 预留插件钩子（lifecycle hooks）供后续系列扩展
 */

import type { Message } from './message.js';
import type { ToolRegistry } from './tool.js';
import type { MemorySystem, MemoryInjector, MemoryInjectionConfig } from './memory.js';
import type { ModelConfig } from './llm.js';
import type { PersistentSession, CheckpointManager, CheckpointTrigger } from './session.js';

/** Agent 配置 */
/** 并行工具执行选项 */
export interface ParallelExecutionOptions {
  /** 最大并行数（默认 5） */
  maxConcurrency?: number;
  /** 单个工具的超时毫秒数（默认 30000） */
  timeoutMs?: number;
  /** 遇到第一个失败就 abort 全部 */
  failFast?: boolean;
  /** 启用依赖冲突检测和分层并行（默认 true） */
  enableConflictDetection?: boolean;
}

/** Agent 配置 */
export interface AgentConfig {
  name: string;
  model: ModelConfig;
  systemPrompt: string;
  tools: ToolRegistry;
  memory: MemorySystem;
  maxIterations?: number;
  /** 并行执行配置。设置后启用并行工具执行，替代默认的串行执行 */
  parallelExecution?: ParallelExecutionOptions;
  /**
   * 记忆注入配置（可选，不配置则不注入长期记忆）
   */
  memoryInjection?: {
    injector: MemoryInjector;
    config?: MemoryInjectionConfig;
  };

  /**
   * 会话配置（可选，不配置则无持久化）
   */
  session?: {
    session: PersistentSession;
    checkpoints?: CheckpointManager;
    checkpointTrigger?: CheckpointTrigger;
  };
}

/** Agent 生命周期钩子 —— 扩展点 */
export interface AgentHooks {
  /** 每次 LLM 调用前触发 */
  onBeforeLLMCall?: (messages: Message[]) => void | Promise<void>;
  /** 每次 LLM 响应后触发 */
  onAfterLLMCall?: (response: unknown) => void | Promise<void>;
  /** 每次 Tool 执行前触发 */
  onBeforeToolCall?: (toolName: string, args: unknown) => void | Promise<void>;
  /** 每次 Tool 执行后触发 */
  onAfterToolCall?: (toolName: string, result: unknown) => void | Promise<void>;
  /** Agent 每次迭代完成时触发 */
  onIterationComplete?: (iteration: number) => void | Promise<void>;
}

/** Agent 执行结果 */
export interface AgentResult {
  messages: Message[];
  /** 最终输出文本 */
  output: string;
  /** 执行了多少轮迭代 */
  iterations: number;
  /** 总 token 用量 */
  totalTokens: number;
  /** 是否因达到 maxIterations 而停止 */
  stoppedEarly: boolean;
}

// ─── 流式输出 ────────────────────────────────────────────────────

/**
 * Agent 流式事件
 *
 * stream() 方法对每个事件类型调用 onEvent 回调。
 * 事件按自然顺序出现：
 *   1. contentDelta（LLM 逐字输出）
 *   2. toolCall（LLM 决定调工具）
 *   3. toolResult（工具执行完成）
 *   4. iteration（一轮迭代完成）
 *   5. done（全部完成，携带最终结果）
 *   6. error（出错终止）
 *
 * contentDelta 和 toolCall 在同一轮迭代中是互斥的——
 * LLM 要么输出文本，要么决定调工具，不会同时发生。
 */
export interface AgentStreamEvent {
  /** LLM 文本输出的增量（逐 token/逐块） */
  contentDelta?: string;
  /** 工具调用信息（LLM 决定调工具时触发） */
  toolCall?: {
    id: string;
    name: string;
    args: Record<string, unknown>;
  };
  /** 工具执行结果 */
  toolResult?: {
    toolCallId: string;
    toolName: string;
    content: string;
    isError: boolean;
  };
  /** 当前迭代轮次 */
  iteration?: number;
  /** 流结束，携带最终 AgentResult */
  done?: AgentResult;
  /** 流式传输中发生错误 */
  error?: string;
}

/**
 * Agent 接口 —— 所有 Agent 实现必须满足此接口。
 * 后续系列可以基于此接口实现特殊 Agent（如 ReAct Agent、Plan-Act Agent 等）。
 */
export interface Agent {
  readonly config: AgentConfig;
  hooks?: AgentHooks;

  /** 执行一次完整对话（支持多轮 tool calling），非流式 */
  run(input: string | Message[]): Promise<AgentResult>;

  /**
   * 流式执行
   *
   * 和 run() 的核心区别：
   * - LLM 的输出不再一次性返回，而是通过 onEvent 逐块推送
   * - 前端可以边接收边渲染，用户看到的是字符逐行出现的效果
   * - 工具调用的开始和结束也有独立事件，方便 UI 展示"正在调用工具"的状态
   *
   * 实现了 Agent 级别的流式输出后，调用方不再需要自己拼 LLM 的流式输出。
   * 只需要注册 onEvent 回调，处理不同类型的事件即可。
   */
  stream(
    input: string | Message[],
    onEvent: (event: AgentStreamEvent) => void
  ): Promise<AgentResult>;

  /** 重置 Agent 状态 */
  reset(): void;
}
