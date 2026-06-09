/**
 * Tool 系统 — Agent 与外部世界交互的桥梁
 *
 * 设计原则：
 * - 每个 Tool 是一个独立模块，有自己的 schema 和执行逻辑
 * - 通过注册机制添加，与 Agent 核心解耦
 * - 后续系列通过实现 Tool 接口就能无缝接入（RedisTool、MySQLTool 等）
 * - 支持 Tool 级别的错误隔离 —— 一个 Tool 挂了不影响其他
 */

import type { z } from 'zod';

/** Tool 元数据 —— 用于 LLM 的 function calling 描述 */
export interface ToolMetadata {
  name: string;
  description: string;
  /** Zod schema，用于参数校验和自动生成 JSON Schema */
  parameters: z.ZodSchema<unknown>;
}

/** Tool 执行上下文 —— 执行时可以获取的信息 */
export interface ToolContext {
  /** 会话 ID（可用于追溯） */
  sessionId?: string;
  /** 附加 metadata */
  metadata?: Record<string, unknown>;
}

/** Tool 执行结果 */
export interface ToolResult {
  success: boolean;
  output: string;
  error?: string;
  /** Tool 可以返回结构化数据供后续处理 */
  data?: unknown;
}

/**
 * Tool 接口 —— 所有工具必须实现此接口
 *
 * 后续新系列只要实现这个接口，注册到 ToolRegistry 即可使用。
 */
export interface Tool {
  metadata: ToolMetadata;

  /** 执行工具逻辑 */
  execute(args: Record<string, unknown>, ctx?: ToolContext): Promise<ToolResult>;
}

/**
 * Tool 注册中心 —— 管理所有可用工具
 *
 * Agent 通过它查找和调用工具，不直接持有工具引用。
 * 这样新系列注册新工具时不需要改动 Agent 核心代码。
 */
export interface ToolRegistry {
  /** 注册一个工具 */
  register(tool: Tool): void;
  /** 批量注册 */
  registerMany(tools: Tool[]): void;
  /** 根据名称获取工具 */
  get(name: string): Tool | undefined;
  /** 列出所有已注册工具的元数据（用于构建 function calling schema） */
  listMetadata(): ToolMetadata[];
  /** 移除工具 */
  unregister(name: string): boolean;
}

/** 默认的 Tool 注册中心实现 */
export class DefaultToolRegistry implements ToolRegistry {
  private tools = new Map<string, Tool>();

  register(tool: Tool): void {
    if (this.tools.has(tool.metadata.name)) {
      throw new Error(`Tool '${tool.metadata.name}' is already registered`);
    }
    this.tools.set(tool.metadata.name, tool);
  }

  registerMany(tools: Tool[]): void {
    for (const tool of tools) {
      this.register(tool);
    }
  }

  get(name: string): Tool | undefined {
    return this.tools.get(name);
  }

  listMetadata(): ToolMetadata[] {
    return Array.from(this.tools.values()).map((t) => t.metadata);
  }

  unregister(name: string): boolean {
    return this.tools.delete(name);
  }
}
