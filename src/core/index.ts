/**
 * 核心模块导出入口
 *
 * 所有核心类型定义集中导出，方便外部模块引用。
 * 后续系列只需要 import { Tool, ToolRegistry } from 'ts-agent-core' 即可。
 */

export type {
  MessageRole,
  ToolCall,
  ToolResult as ToolCallResult,
  Message,
  ConversationContext,
} from './message.js';
export {
  systemMessage,
  userMessage,
  assistantMessage,
  toolResultMessage,
} from './message.js';

export type {
  ToolMetadata,
  ToolContext,
  ToolResult,
  Tool,
  ToolRegistry,
} from './tool.js';
export { DefaultToolRegistry } from './tool.js';

export type {
  ShortTermMemory,
  LongTermMemoryItem,
  LongTermMemoryQuery,
  LongTermMemory,
  MemorySystem,
} from './memory.js';
export { SlidingWindowMemory } from './memory.js';

export type {
  LLMRequest,
  LLMResponse,
  LLMChunk,
  LLMProvider,
  ModelConfig,
} from './llm.js';

export type {
  AgentConfig,
  AgentHooks,
  AgentResult,
  Agent,
} from './agent.js';

export type {
  Session,
  SessionCheckpoint,
} from './session.js';
export { InMemorySession } from './session.js';
