/**
 * 消息模块 — Agent 对话的基本单元
 *
 * 设计原则：
 * - 兼容 OpenAI/Anthropic 等多厂商的消息格式
 * - 预留 tool_call / tool_result 两种特殊角色
 * - 与具体的 LLM Provider 解耦（Provider 负责转换成本地格式）
 */

/** 消息角色 */
export type MessageRole = 'system' | 'user' | 'assistant' | 'tool';

/** Tool 调用请求 */
export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

/** Tool 调用结果 */
export interface ToolResult {
  id: string;
  name: string;
  content: string;
  isError?: boolean;
}

/** 一条完整消息 */
export interface Message {
  role: MessageRole;
  content: string;
  toolCalls?: ToolCall[];
  toolCallId?: string;
  toolName?: string;
  isError?: boolean;
  metadata?: Record<string, unknown>;
}

/** 对话上下文 —— 消息列表的封装，方便做 token 预算控制 */
export interface ConversationContext {
  messages: Message[];
  /** 预估的 token 总数（由 Tokenizer 或 Provider 填充） */
  estimatedTokens?: number;
  /** 系统提示词（独立字段，方便后续做 prompt 优化压缩） */
  systemPrompt?: string;
}

/** 构建消息的工厂函数 */
export function systemMessage(content: string): Message {
  return { role: 'system', content };
}

export function userMessage(content: string): Message {
  return { role: 'user', content };
}

export function assistantMessage(
  content: string,
  toolCalls?: ToolCall[]
): Message {
  return { role: 'assistant', content, toolCalls };
}

export function toolResultMessage(
  toolCallId: string,
  toolName: string,
  content: string,
  isError?: boolean
): Message {
  return { role: 'tool', content, toolCallId, toolName, isError };
}
