/**
 * Provider 消息适配器
 *
 * 将统一的消息模型（Message）转换为不同 LLM Provider 的专有格式。
 *
 * 设计原则：
 * - 核心消息模型与 Provider 解耦，不引入任何外部 SDK 依赖
 * - 每个 Provider 的适配器独立，新增 Provider 只需加一个转换函数
 * - 格式转换是纯函数——不依赖状态，便于测试
 */

import type { Message, ToolCall } from './message.js';

// ─── OpenAI 格式 ───────────────────────────────────────────────

/**
 * OpenAI Chat Completion message 格式
 *
 * 参考: https://platform.openai.com/docs/api-reference/chat/create
 */
export interface OpenAIMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: Array<{
    id: string;
    type: 'function';
    function: {
      name: string;
      arguments: string;
    };
  }>;
  tool_call_id?: string;
  name?: string;
}

/**
 * 将内部 Message 转换为 OpenAI 格式
 */
export function toOpenAIMessages(messages: Message[]): OpenAIMessage[] {
  return messages.map(toOpenAIMessage);
}

function toOpenAIMessage(msg: Message): OpenAIMessage {
  const base: OpenAIMessage = {
    role: msg.role,
    content: msg.content || null,
  };

  if (msg.role === 'assistant' && msg.toolCalls && msg.toolCalls.length > 0) {
    base.tool_calls = msg.toolCalls.map((tc) => ({
      id: tc.id,
      type: 'function' as const,
      function: {
        name: tc.name,
        arguments: JSON.stringify(tc.args),
      },
    }));
  }

  if (msg.role === 'tool') {
    base.tool_call_id = msg.toolCallId;
    base.name = msg.toolName;
  }

  return base;
}

/**
 * 将 OpenAI 的 tool_call 格式转换为内部 ToolCall
 */
export function fromOpenAIToolCalls(
  toolCalls?: OpenAIMessage['tool_calls']
): ToolCall[] | undefined {
  if (!toolCalls || toolCalls.length === 0) return undefined;

  return toolCalls.map((tc) => ({
    id: tc.id,
    name: tc.function.name,
    args: safeParseJSON(tc.function.arguments),
  }));
}

// ─── Anthropic 格式 ────────────────────────────────────────────

/**
 * Anthropic Messages API content block
 *
 * 参考: https://docs.anthropic.com/en/api/messages
 */
export type AnthropicContent =
  | { type: 'text'; text: string }
  | { type: 'tool_use'; id: string; name: string; input: Record<string, unknown> }
  | { type: 'tool_result'; tool_use_id: string; content: string; is_error?: boolean };

export interface AnthropicMessage {
  role: 'user' | 'assistant';
  content: AnthropicContent[];
}

/**
 * 将内部 Message 列表转换为 Anthropic Messages API 格式
 *
 * Anthropic 的格式和 OpenAI 有一个关键区别：
 * - tool_result 属于 user role（Anthropic 认为工具结果是用户的上下文）
 * - tool_use 属于 assistant role
 * - 不支持 system role 在 messages 数组中，system 通过独立参数传递
 */
export function toAnthropicMessages(
  messages: Message[],
  options?: { separateSystem?: boolean }
): AnthropicMessage[] {
  // 提取 system prompt
  const systemMessages = messages.filter((m) => m.role === 'system');
  const nonSystemMessages = messages.filter((m) => m.role !== 'system');

  if (options?.separateSystem === false) {
    // 不分离 system，合并到 messages（部分 Anthropic 兼容 API 支持）
    return convertToAnthropic(messages);
  }

  return convertToAnthropic(nonSystemMessages);
}

function convertToAnthropic(msgs: Message[]): AnthropicMessage[] {
  const result: AnthropicMessage[] = [];
  let currentBlock: AnthropicContent[] = [];
  let currentRole: 'user' | 'assistant' | null = null;

  function flush() {
    if (currentBlock.length > 0 && currentRole) {
      result.push({ role: currentRole, content: currentBlock });
      currentBlock = [];
      currentRole = null;
    }
  }

  for (const msg of msgs) {
    if (msg.role === 'user') {
      flush();
      currentRole = 'user';
      currentBlock.push({ type: 'text', text: msg.content });
    } else if (msg.role === 'assistant') {
      flush();
      currentRole = 'assistant';
      if (msg.content) {
        currentBlock.push({ type: 'text', text: msg.content });
      }
      if (msg.toolCalls) {
        for (const tc of msg.toolCalls) {
          currentBlock.push({
            type: 'tool_use',
            id: tc.id,
            name: tc.name,
            input: tc.args as Record<string, unknown>,
          });
        }
      }
    } else if (msg.role === 'tool') {
      // Anthropic: tool_result 的内容属于 user（这是 Anthropic API 的强制约定）
      // 注意：连续多个 tool_result 会合并到同一个 user 消息的 content 数组中
      // Anthropic API 允许一个 user 消息中有多个 tool_result block，这是合规的
      if (currentRole !== 'user') {
        flush();
        currentRole = 'user';
      }
      currentBlock.push({
        type: 'tool_result',
        tool_use_id: msg.toolCallId ?? '',
        content: msg.content,
        is_error: msg.isError,
      });
    } else if (msg.role === 'system') {
      // system 消息一般不放在 messages 数组中
      // 但如果用户选择不分离，这里仍然可以处理
      flush();
      currentRole = 'user';
      currentBlock.push({ type: 'text', text: msg.content });
    }
  }

  flush();
  return result;
}

/**
 * 从 Anthropic Messages API 响应中提取 system prompt
 */
export function extractAnthropicSystemPrompt(
  messages: Message[]
): string | undefined {
  const systemMessages = messages.filter((m) => m.role === 'system');
  if (systemMessages.length === 0) return undefined;

  // Anthropic 的 system prompt 是一个独立参数，多条 system 消息合并
  return systemMessages.map((m) => m.content).join('\n');
}

// ─── 工具函数 ────────────────────────────────────────────────────

function safeParseJSON(text: string): Record<string, unknown> {
  try {
    return JSON.parse(text) as Record<string, unknown>;
  } catch {
    return {};
  }
}
