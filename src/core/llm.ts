/**
 * LLM Provider 抽象 — 屏蔽不同 LLM 厂商的 API 差异
 *
 * 设计原则：
 * - 统一的 Completions 接口
 * - Tool calling 支持（不强制要求所有 Provider 都支持）
 * - 流式输出预留
 * - 通过接口实现可切换后端（OpenAI、Anthropic、本地模型等）
 */

import type { Message } from './message.js';
import type { ToolMetadata } from './tool.js';

/** LLM 请求参数 */
export interface LLMRequest {
  messages: Message[];
  /** 模型名称（如 'gpt-4o'、'claude-3-5-sonnet'），Provider 用此选择后端 */
  model?: string;
  tools?: ToolMetadata[];
  /** tool_choice，可选 */
  toolChoice?: 'auto' | 'none' | 'required';
  /** 模型参数 */
  maxTokens?: number;
  temperature?: number;
  /** 额外的厂商特定参数（通过这个字段向下透传，不阻塞扩展） */
  extraParams?: Record<string, unknown>;
}

/** LLM 响应 */
export interface LLMResponse {
  content: string;
  toolCalls?: Array<{ id: string; name: string; args: Record<string, unknown> }>;
  /** 原始响应中透传的信息（用量、延迟等） */
  usage?: {
    promptTokens: number;
    completionTokens: number;
    totalTokens: number;
  };
  finishReason: 'stop' | 'tool_calls' | 'length' | 'error';
}

/** 流式响应的一块 */
export interface LLMChunk {
  contentDelta?: string;
  toolCallDelta?: Partial<{ id: string; name: string; args: string }>;
  finishReason?: 'stop' | 'tool_calls' | 'length' | 'error';
}

/** LLM Provider 接口 */
export interface LLMProvider {
  readonly name: string;
  /** 非流式调用 */
  complete(request: LLMRequest): Promise<LLMResponse>;
  /** 流式调用，逐块回调 */
  completeStream(
    request: LLMRequest,
    onChunk: (chunk: LLMChunk) => void
  ): Promise<LLMResponse>;
}

/** 模型配置——用于在 Agent 初始化时指定 LLM 后端 */
export interface ModelConfig {
  provider: LLMProvider;
  model: string;
  maxTokens?: number;
  temperature?: number;
}
