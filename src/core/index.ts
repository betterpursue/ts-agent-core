/**
 * 核心模块导出入口
 *
 * 所有核心类型定义集中导出，方便外部模块引用。
 * 后续系列只需要 import { Tool, ToolRegistry } from 'ts-agent-core' 即可。
 */

export type {
  MessageRole,
  ToolCall,

  Message,
  ConversationContext,
  MessageValidation,
  TrimStrategy,
} from './message.js';
export {
  systemMessage,
  userMessage,
  assistantMessage,
  toolResultMessage,
  validateMessage,
  validateConversation,
  TokenEstimator,
  defaultTokenEstimator,
  MessageWindow,
} from './message.js';

export type {
  ToolMetadata,
  ToolContext,
  ToolResult,
  ToolDependency,
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
  AgentStreamEvent,
  Agent,
  ParallelExecutionOptions,
} from './agent.js';

export type {
  Session,
  SessionCheckpoint,
  PersistentSession,
  CheckpointManager,
  CheckpointTrigger,
} from './session.js';
export {
  InMemorySession,
  FileSession,
  FileCheckpointManager,
  RetentionCheckpointManager,
} from './session.js';

export type {
  OpenAIMessage,
  AnthropicMessage,
  AnthropicContent,
} from './provider-adapter.js';
export {
  toOpenAIMessages,
  fromOpenAIToolCalls,
  toAnthropicMessages,
  extractAnthropicSystemPrompt,
} from './provider-adapter.js';
