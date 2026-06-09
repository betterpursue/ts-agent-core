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
import type { MemorySystem } from './memory.js';
import type { ModelConfig } from './llm.js';

/** Agent 配置 */
export interface AgentConfig {
  name: string;
  model: ModelConfig;
  systemPrompt: string;
  tools: ToolRegistry;
  memory: MemorySystem;
  maxIterations?: number; // 单次执行最大循环轮数（防止死循环）
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

/**
 * Agent 接口 —— 所有 Agent 实现必须满足此接口。
 * 后续系列可以基于此接口实现特殊 Agent（如 ReAct Agent、Plan-Act Agent 等）。
 */
export interface Agent {
  readonly config: AgentConfig;
  hooks?: AgentHooks;

  /** 执行一次完整对话（支持多轮 tool calling） */
  run(input: string | Message[]): Promise<AgentResult>;

  /** 重置 Agent 状态 */
  reset(): void;
}
