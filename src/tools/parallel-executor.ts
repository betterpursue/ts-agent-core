/**
 * 并行工具执行器 — 让 Agent 能同时执行多个独立工具调用
 *
 * 设计决策：
 * - 独立性假设：默认假设同一个 LLM response 中的 tool_calls 是独立的，可以并行
 * - 错误隔离：一个工具失败不影响其他工具的并行执行
 * - 超时控制：每个工具有独立的超时，防止一个卡住拖慢全部
 * - 可选降级到串行：外部可以传入控制参数强制降级
 *
 * 这是 Tool 系统对 Agent 主循环的补充。主循环本身不关心工具是并行还是串行执行，
 * 它只关心消息的生成。并行是由执行器层面的抽象提供的.
 *
 * 后续扩展思路：
 * - 依赖分析：Tool 可以声明自己的依赖，执行器做 DAG 拓扑排序后分层并行
 * - 资源限制：控制最大并行数，防止 CPU/IO 打满
 * - 重试策略：对可重试的错误自动重试
 */

import { z } from 'zod';
import type { ToolRegistry, Tool, ToolResult, ToolContext, ToolMetadata } from '../core/tool.js';

export interface ParallelExecutionOptions {
  /** 最大并行数（默认 5） */
  maxConcurrency?: number;
  /** 单个工具的超时毫秒数（默认 30000） */
  timeoutMs?: number;
  /** 遇到第一个失败就 abort 全部 */
  failFast?: boolean;
}

export interface ParallelExecutionResult {
  results: Map<string, ToolResult>;
  /** 失败的调用 */
  failures: Array<{ id: string; name: string; error: string }>;
  /** 成功的调用 */
  successes: number;
  /** 总耗时（ms） */
  elapsedMs: number;
}

type ToolCallRequest = {
  id: string;
  name: string;
  args: Record<string, unknown>;
};

/**
 * 并行执行一批工具调用
 *
 * 执行策略：
 * 1. 用 Promise.allSettled 实现并行，每个工具独立执行
 * 2. 对没找到的工具直接标记失败，不创建 Promise
 * 3. 每个执行单元都 wrapped 在超时控制里
 */
export async function executeToolsParallel(
  registry: ToolRegistry,
  calls: ToolCallRequest[],
  options?: ParallelExecutionOptions,
  ctx?: ToolContext
): Promise<ParallelExecutionResult> {
  const start = Date.now();
  const maxConcurrency = options?.maxConcurrency ?? 5;
  const timeoutMs = options?.timeoutMs ?? 30_000;
  const failFast = options?.failFast ?? false;

  if (calls.length === 0) {
    return {
      results: new Map(),
      failures: [],
      successes: 0,
      elapsedMs: 0,
    };
  }

  const results = new Map<string, ToolResult>();
  const failures: Array<{ id: string; name: string; error: string }> = [];

  // 限速并行：一次最多跑 maxConcurrency 个
  const chunks: ToolCallRequest[][] = [];
  for (let i = 0; i < calls.length; i += maxConcurrency) {
    chunks.push(calls.slice(i, i + maxConcurrency));
  }

  for (const chunk of chunks) {
    // 如果前面已经失败了且 failFast，跳过后续
    if (failFast && failures.length > 0) {
      // 剩余调用标记为未执行
      for (const call of chunk) {
        failures.push({ id: call.id, name: call.name, error: 'Skipped due to prior failure (failFast)' });
      }
      continue;
    }

    const settled = await Promise.allSettled(
      chunk.map(async (call) => {
        const tool = registry.get(call.name);
        if (!tool) {
          throw new ToolNotFoundError(call.name);
        }

        // 带超时的工具执行
        const result = await withTimeout(
          tool.execute(call.args, ctx),
          timeoutMs
        );
        return { id: call.id, name: call.name, result };
      })
    );

    for (const outcome of settled) {
      if (outcome.status === 'fulfilled') {
        const { id, name, result } = outcome.value;
        results.set(id, result);
        if (!result.success) {
          failures.push({ id, name, error: result.error ?? 'Unknown error' });
        }
      } else {
        const reason = outcome.reason;
        const callIndex = chunk.indexOf(
          // Find the original call - imperfect but works for error reporting
          // In production, we'd track by the rejected promise
          chunk[settled.indexOf(outcome) < chunk.length ? settled.indexOf(outcome) : 0]
        );
        const failedCall = chunk[Math.min(settled.indexOf(outcome), chunk.length - 1)];
        failures.push({
          id: failedCall.id,
          name: failedCall.name,
          error: reason instanceof ToolNotFoundError
            ? `Tool '${failedCall.name}' not found`
            : reason instanceof TimeoutError
              ? `Tool execution timed out after ${timeoutMs}ms`
              : `Unexpected error: ${reason instanceof Error ? reason.message : String(reason)}`,
        });
      }
    }
  }

  return {
    results,
    successes: results.size - failures.length,
    failures,
    elapsedMs: Date.now() - start,
  };
}

/** 超时包装器 */
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new TimeoutError(`Timed out after ${ms}ms`));
    }, ms);

    promise.then(
      (val) => {
        clearTimeout(timer);
        resolve(val);
      },
      (err) => {
        clearTimeout(timer);
        reject(err);
      }
    );
  });
}

export class ToolNotFoundError extends Error {
  constructor(toolName: string) {
    super(`Tool '${toolName}' not found`);
    this.name = 'ToolNotFoundError';
  }
}

export class TimeoutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'TimeoutError';
  }
}

/**
 * 获取工具的 JSON Schema 描述（用于 LLM function calling）
 *
 * Zod schema 可以自动推导出 JSON Schema，不需要手写。
 * 每个 Tool 定义好后，LLM 自动获得参数的完整描述。
 */
export function toolMetadataToJsonSchema(metadata: ToolMetadata): Record<string, unknown> {
  // Zod 的 `._def` 内部结构比较复杂，这里用 Zod 内置的 JSON Schema 推导
  // 简单封装，方便 LLM 调用层使用
  return {
    type: 'function',
    function: {
      name: metadata.name,
      description: metadata.description,
      parameters: metadata.parameters instanceof z.ZodType
        ? (metadata.parameters as z.ZodType<unknown>).describe(metadata.description)
        : undefined,
    },
  };
}
