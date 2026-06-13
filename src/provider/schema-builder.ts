/**
 * Function Calling Schema 构建器
 *
 * 将 ToolMetadata 转换为 LLM API 可理解的 function calling 描述。
 * 核心职责：把我们的 Zod-based Tool 定义变成 OpenAI / Anthropic 能读的 JSON Schema。
 *
 * 设计原则：
 * - 直接使用 Zod v4 的 toJSONSchema()，不需要手写 Zod → JSON Schema 的转换
 * - 每个 Tool 只生成一份 schema，可缓存（避免重复转换）
 * - 支持 OpenAI 和 Anthropic 两种格式
 * - 参数校验：在交给 LLM 之前就能发现 schema 定义错误
 */

import { z } from 'zod';
import type { ToolMetadata } from '../core/tool.js';

// ─── OpenAI Function Calling Schema ─────────────────────────────

/**
 * OpenAI function calling 的 function 描述
 *
 * https://platform.openai.com/docs/api-reference/chat/create#chat-create-tools
 */
export interface OpenAIFunction {
  name: string;
  description: string;
  parameters: Record<string, unknown>;
  /** strict 模式下，OpenAI 会强制按 schema 输出 */
  strict?: boolean;
}

/**
 * OpenAI function calling 的完整 tool 描述
 */
export interface OpenAITool {
  type: 'function';
  function: OpenAIFunction;
}

// ─── Schema 构建 ────────────────────────────────────────────────

/**
 * 从 Zod v4 schema 直接生成 OpenAI 兼容的 JSON Schema
 *
 * Zod v4 内置了 toJSONSchema() 方法，生成的 JSON Schema 符合
 * JSON Schema 2020-12 规范，可以直接喂给 OpenAI API。
 *
 * 处理方法：
 * - 对于 ZodObject：直接调用 toJSONSchema()
 * - 对于其他类型：用 z.object({value: schema}) 包装后再调用
 */
function zodToJsonSchema(schema: z.ZodType<unknown>): Record<string, unknown> {
  const jsonSchema = schema.toJSONSchema() as Record<string, unknown>;

  // 移除 $schema 字段（OpenAI 不需要）
  if (jsonSchema['$schema']) {
    delete jsonSchema['$schema'];
  }

  // 移除 additionalProperties（OpenAI 的 strict mode 会自动处理）
  if ('additionalProperties' in jsonSchema) {
    delete jsonSchema['additionalProperties'];
  }

  return jsonSchema;
}

/**
 * 将单个 ToolMetadata 转换为 OpenAI tool 描述
 *
 * @param metadata - Tool 的元数据
 * @param strict  - 是否启用 strict mode（默认 false）
 *   strict mode 下，OpenAI 会强制 LLM 按 schema 输出参数，
 *   但要求所有参数都是 required。
 */
export function buildOpenAITool(
  metadata: ToolMetadata,
  strict = false
): OpenAITool {
  const parameters = zodToJsonSchema(metadata.parameters);

  // strict mode 下，OpenAI 要求所有参数都是 required
  if (strict) {
    const params = parameters as {
      properties?: Record<string, unknown>;
      required?: string[];
    };
    if (params.properties) {
      // 覆盖 required，确保全部字段都设为必填
      params.required = Object.keys(params.properties);
    }
  }

  const func: OpenAIFunction = {
    name: metadata.name,
    description: metadata.description,
    parameters,
  };

  if (strict) {
    func.strict = true;
  }

  return {
    type: 'function',
    function: func,
  };
}

/**
 * 批量构建 OpenAI tools 数组
 *
 * @param metadatas - 所有已注册 Tool 的元数据列表
 * @param strict    - 是否启用 strict mode
 */
export function buildOpenAITools(
  metadatas: ToolMetadata[],
  strict = false
): OpenAITool[] {
  return metadatas.map((m) => buildOpenAITool(m, strict));
}

/**
 * 校验 Tool 的 schema 是否可用
 *
 * 在启动时做一次校验，防止运行时才发现 schema 定义有问题。
 * 常见的 schema 问题：
 * - 没有定义 parameters（至少需要一个参数对象）
 * - description 为空
 */
export function validateToolSchema(metadata: ToolMetadata): {
  valid: boolean;
  errors: string[];
} {
  const errors: string[] = [];

  if (!metadata.name || metadata.name.trim().length === 0) {
    errors.push('Tool name is required');
  }

  if (!metadata.description || metadata.description.trim().length === 0) {
    errors.push(`Tool '${metadata.name}' has no description`);
  }

  if (!metadata.parameters) {
    errors.push(`Tool '${metadata.name}' has no parameters schema`);
  } else {
    // 尝试转换，捕获 Zod schema 中的异常
    try {
      zodToJsonSchema(metadata.parameters);
    } catch (err) {
      errors.push(
        `Tool '${metadata.name}' schema conversion failed: ${err instanceof Error ? err.message : String(err)}`
      );
    }
  }

  return { valid: errors.length === 0, errors };
}

// ─── Anthropic Tool Schema ──────────────────────────────────────

/**
 * Anthropic Tool 描述格式
 *
 * https://docs.anthropic.com/en/docs/build-with-claude/tool-use
 */
export interface AnthropicTool {
  name: string;
  description: string;
  input_schema: Record<string, unknown>;
}

/**
 * 将 ToolMetadata 转换为 Anthropic tool 描述
 *
 * Anthropic 和 OpenAI 的 tool schema 区别不大：
 * - 没有 type: 'function' 包装
 * - 不需要 strict 模式
 * - input_schema 直接暴露（不是嵌套在 parameters 里）
 */
export function buildAnthropicTool(metadata: ToolMetadata): AnthropicTool {
  return {
    name: metadata.name,
    description: metadata.description,
    input_schema: zodToJsonSchema(metadata.parameters),
  };
}

export function buildAnthropicTools(metadatas: ToolMetadata[]): AnthropicTool[] {
  return metadatas.map(buildAnthropicTool);
}

// ─── Schema 缓存 ────────────────────────────────────────────────

/**
 * Schema 缓存 —— 避免重复转换
 *
 * Tool 一旦注册就不会变，所以 schema 转换结果可以安全缓存。
 * 如果后续支持热更新 Tool，需要提供一个 invalidate 方法。
 */
export class SchemaCache {
  private openAICache = new Map<string, OpenAITool>();
  private anthropicCache = new Map<string, AnthropicTool>();
  private strict = false;

  constructor(strict = false) {
    this.strict = strict;
  }

  /** 获取或创建 OpenAI tool schema */
  getOrCreateOpenAI(metadata: ToolMetadata): OpenAITool {
    const cached = this.openAICache.get(metadata.name);
    if (cached) return cached;

    const tool = buildOpenAITool(metadata, this.strict);
    this.openAICache.set(metadata.name, tool);
    return tool;
  }

  /** 批量获取 OpenAI schema */
  getOpenAITools(metadatas: ToolMetadata[]): OpenAITool[] {
    return metadatas.map((m) => this.getOrCreateOpenAI(m));
  }

  /** 获取或创建 Anthropic tool schema */
  getOrCreateAnthropic(metadata: ToolMetadata): AnthropicTool {
    const cached = this.anthropicCache.get(metadata.name);
    if (cached) return cached;

    const tool = buildAnthropicTool(metadata);
    this.anthropicCache.set(metadata.name, tool);
    return tool;
  }

  /** 批量获取 Anthropic schema */
  getAnthropicTools(metadatas: ToolMetadata[]): AnthropicTool[] {
    return metadatas.map((m) => this.getOrCreateAnthropic(m));
  }

  /** 清除缓存（当 Tool 热更新时调用） */
  invalidate(): void {
    this.openAICache.clear();
    this.anthropicCache.clear();
  }
}

/** 默认的 schema 缓存实例 */
export const defaultSchemaCache = new SchemaCache();
