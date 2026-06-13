/**
 * Provider 模块导出
 *
 * Function Calling 链路的核心组件：Schema 构建 + 响应解析 + Provider 实现。
 */

export type { OpenAIFunction, OpenAITool, AnthropicTool } from './schema-builder.js';
export {
  buildOpenAITool,
  buildOpenAITools,
  buildAnthropicTool,
  buildAnthropicTools,
  validateToolSchema,
  SchemaCache,
  defaultSchemaCache,
} from './schema-builder.js';

export type { ParsedToolCall, ParseResult } from './response-parser.js';
export {
  repairJson,
  parseToolCalls,
  buildToolResultMessage,
  parseResultSummary,
} from './response-parser.js';

export type { OpenAIProviderConfig } from './openai-provider.js';
export {
  OpenAIProvider,
  OpenAIError,
  OpenAIRateLimitError,
  OpenAIAuthError,
} from './openai-provider.js';
