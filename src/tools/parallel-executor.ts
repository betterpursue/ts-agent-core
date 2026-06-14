/**
 * 并行工具执行器 — 让 Agent 同时执行多个独立工具调用
 *
 * 设计决策：
 * =========
 *
 * 1. 为什么不做隐式依赖检测？
 *    LLM 的 tool_call 参数在执行前是黑盒，无法静态分析工具会触及哪些共享资源。
 *    所以我们要求工具声明依赖（ToolDependency），运行时做冲突检测。
 *    这是显式声明优于隐式推断的经典 trade-off：多一行声明，换来确定性的并行安全。
 *
 * 2. 为什么用 Semaphore 而不是 Promise.allSettled + chunk？
 *    旧版 parallel-executor.ts 把 calls 切块，每块 Promise.allSettled。问题是：
 *    - 如果块大小=5，块内所有工具都完成后才执行下一批，不能动态填槽
 *    - 一个慢工具拖慢整块
 *    Semaphore 模型允许"执行完一个立刻填下一个"，更高效。
 *
 * 3. 竞态处理策略：依赖分组 + 分层并行
 *    把有冲突的工具串行化（同一层内无冲突，层间串行），
 *    无冲突的工具任意并行。这是数据库隔离级别中"可重复读"的朴素实现。
 *
 * 4. 错误隔离
 *    一个工具失败不影响其他同层工具的执行。
 *    只有 failFast 模式开启时，首错即停。
 *
 * 后续扩展思路：
 * - wait-die / wound-wait 死锁预防
 * - 两阶段锁定（2PL）：工具执行前锁定所有 writes，执行后释放
 * - 可撤销工具：工具声明为可撤销，发生冲突时回滚重试
 */

import type { ToolRegistry, Tool, ToolResult, ToolContext, ToolDependency } from '../core/tool.js';
import type { ParallelExecutionOptions } from '../core/agent.js';

// ─── Semaphore（信号量） ────────────────────────────────────────

/**
 * Semaphore — 并发度控制器
 *
 * 相比 chunk-based 方案，Semaphore 能更平滑地控制并发：
 * - 整体执行时间 ≈ 最长执行时间 * 层数，而不是 块大小 * 层数
 * - 不会因为块内有一个慢任务堵住后面的快任务
 */
class Semaphore {
  private count: number;
  private queue: Array<() => void> = [];

  constructor(private maxConcurrency: number) {
    this.count = maxConcurrency;
  }

  /** 获取一个执行槽（没有就排队） */
  acquire(): Promise<void> {
    if (this.count > 0) {
      this.count--;
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      this.queue.push(resolve);
    });
  }

  /** 释放一个执行槽 */
  release(): void {
    if (this.queue.length > 0) {
      // 有排队的，直接让给队首
      const next = this.queue.shift()!;
      next();
    } else {
      this.count++;
    }
  }

  /** 在信号量保护下执行异步函数 */
  async run<T>(fn: () => Promise<T>): Promise<T> {
    await this.acquire();
    try {
      return await fn();
    } finally {
      this.release();
    }
  }
}

// ─── 超时控制 ────────────────────────────────────────────────────

class TimeoutError extends Error {
  constructor(ms: number) {
    super(`Tool execution timed out after ${ms}ms`);
    this.name = 'TimeoutError';
  }
}

class ToolNotFoundError extends Error {
  constructor(toolName: string) {
    super(`Tool '${toolName}' not found`);
    this.name = 'ToolNotFoundError';
  }
}

// ─── 依赖解析 ────────────────────────────────────────────────────

/**
 * 冲突类型
 */
type ConflictType = 'read-write' | 'write-read' | 'write-write';

/**
 * 两张工具调用是否存在冲突
 *
 * 冲突规则（与读写锁一致）：
 * - 两个工具如果都只读同一个 key：不冲突
 * - 任何工具写了一个 key，另一个工具读或写同一 key：冲突
 */
function hasConflict(a: ToolDependency, b: ToolDependency): boolean {
  const aReads = new Set(a.reads ?? []);
  const aWrites = new Set(a.writes ?? []);
  const bReads = new Set(b.reads ?? []);
  const bWrites = new Set(b.writes ?? []);

  // 检查 a 的写入是否被 b 读到
  for (const key of aWrites) {
    if (bReads.has(key) || bWrites.has(key)) return true;
  }

  // 检查 a 的读取是否被 b 写入
  for (const key of aReads) {
    if (bWrites.has(key)) return true;
  }

  return false;
}

/**
 * 将一批工具调用分层，使同一层内的工具无冲突，可以并行执行
 *
 * 算法：贪心分层
 * 1. 从剩余调用中，取出一个子集，使子集内两两无冲突
 * 2. 将这个子集作为一个层（layer）
 * 3. 重复直到所有调用都被分配
 *
 * 这不是最优分层（最小层数），但简单、确定、对工具数量 < 20 的场景足够。
 * 如果未来有大规模 DAG 需求，可以替换为最大团分解算法。
 */
function buildLayers(
  calls: Array<{ id: string; name: string; dependency: ToolDependency }>
): Array<Array<{ id: string; name: string; dependency: ToolDependency }>> {
  const layers: typeof calls[] = [];
  const remaining = [...calls];

  while (remaining.length > 0) {
    const layer: typeof calls = [];
    const toRemove: number[] = [];

    for (let i = 0; i < remaining.length; i++) {
      const call = remaining[i];
      const conflictsWithLayer = layer.some((l) => hasConflict(l.dependency, call.dependency));

      if (!conflictsWithLayer) {
        layer.push(call);
        toRemove.push(i);
      }
    }

    // 从后往前删除，保持索引正确
    for (let i = toRemove.length - 1; i >= 0; i--) {
      remaining.splice(toRemove[i], 1);
    }

    layers.push(layer);
  }

  return layers;
}

// ─── 结果类型 ──────────────────────────────────────────────────

export interface ParallelExecutionResult {
  results: Map<string, ToolResult>;
  /** 失败的调用 */
  failures: Array<{ id: string; name: string; error: string }>;
  /** 成功执行了多少工具 */
  successes: number;
  /** 总共耗时（ms） */
  elapsedMs: number;
  /** 执行的层数（依赖分析后分层总数） */
  layers: number;
}

export type ToolCallRequest = {
  id: string;
  name: string;
  args: Record<string, unknown>;
};

// ─── 主执行函数 ──────────────────────────────────────────────────

/**
 * 并行执行一批工具调用
 *
 * 执行策略：
 * 1. 如果 enableConflictDetection，先做依赖分析、分层
 * 2. 每层内的工具通过 Semaphore 并发执行
 * 3. 层与层之间串行执行
 * 4. 每个工具独立超时控制
 * 5. 错误隔离：一个工具失败不影响同层其他工具
 *
 * 旧版 parallel-executor.ts 的缺陷修复：
 * - 不再用 Arrays.splice 和 settled.indexOf(outcome) 这种不可靠的匹配方式
 * - 每个 Promise 自带 id，解析后直接放回 Map
 * - 超时控制在每个工具级别，不在块级别
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
  const detectConflicts = options?.enableConflictDetection ?? true;

  if (calls.length === 0) {
    return {
      results: new Map(),
      failures: [],
      successes: 0,
      elapsedMs: 0,
      layers: 0,
    };
  }

  const results = new Map<string, ToolResult>();
  const failures: Array<{ id: string; name: string; error: string }> = [];
  const semaphore = new Semaphore(maxConcurrency);

  // 1. 构建依赖信息
  const callDeps = calls.map((call) => {
    const tool = registry.get(call.name);
    const dependency: ToolDependency = tool?.dependencies ?? {};
    return { id: call.id, name: call.name, args: call.args, dependency };
  });

  // 2. 如果启用冲突检测，做依赖分析分层
  const layers = detectConflicts
    ? buildLayers(callDeps.map(({ id, name, dependency }) => ({ id, name, dependency })))
    // 不检测冲突：全部放到一层，最大并行
    : [callDeps.map(({ id, name, dependency }) => ({ id, name, dependency }))];

  // 3. 逐层执行
  for (const layer of layers) {
    // failFast 模式：先检查是否已有失败
    if (failFast && failures.length > 0) {
      for (const call of layer) {
        failures.push({ id: call.id, name: call.name, error: 'Skipped due to prior failure (failFast)' });
      }
      continue;
    }

    // 找到这一层对应的完整调用信息（带 args）
    const layerCallIds = new Set(layer.map((c) => c.id));
    const layerCalls = callDeps.filter((c) => layerCallIds.has(c.id));

    // 用 Semaphore 控制并发，每个工具独立执行
    //
    // 设计决策：所有执行结果都用 fulfilled Promise 传递，不依赖 rejection 做错误传播。
    // 好处是 Promise.allSettled 后不需要判断 status，统一处理 value 对象。
    // 每个 call 要么返回 { ok: true, result }，要么返回 { ok: false, error }。
    const layerPromises = layerCalls.map((call) =>
      semaphore.run(async (): Promise<{
        id: string;
        name: string;
        ok: true;
        result: ToolResult;
      } | {
        id: string;
        name: string;
        ok: false;
        error: string;
      }> => {
        const tool = registry.get(call.name);
        if (!tool) {
          return { id: call.id, name: call.name, ok: false, error: `Tool '${call.name}' not found` };
        }

        try {
          const result = await withTimeout(tool.execute(call.args, ctx), timeoutMs);
          return { id: call.id, name: call.name, ok: true, result };
        } catch (err) {
          if (err instanceof TimeoutError) {
            return { id: call.id, name: call.name, ok: false, error: err.message };
          }
          const msg = err instanceof Error ? err.message : String(err);
          return { id: call.id, name: call.name, ok: false, error: `Unexpected error: ${msg}` };
        }
      })
    );

    const outcomes = await Promise.all(layerPromises);

    // 所有 Promise 都会 fulfilled，统一处理 value
    for (const outcome of outcomes) {
      if (outcome.ok) {
        results.set(outcome.id, outcome.result);
        if (!outcome.result.success) {
          failures.push({ id: outcome.id, name: outcome.name, error: outcome.result.error ?? 'Unknown error' });
        }
      } else {
        failures.push({ id: outcome.id, name: outcome.name, error: outcome.error });
      }
    }
  }

  // successes = 成功执行的工具数 = 在 results 中且不在 failures 里的
  const failedIds = new Set(failures.map((f) => f.id));
  const successCount = Array.from(results.entries()).filter(
    ([id, r]) => r.success && !failedIds.has(id)
  ).length;

  return {
    results,
    successes: successCount,
    failures,
    elapsedMs: Date.now() - start,
    layers: layers.length,
  };
}

// ─── 内部辅助 ────────────────────────────────────────────────────

/** 带超时的 Promise 包装 */
function withTimeout<T>(promise: Promise<T>, ms: number): Promise<T> {
  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      reject(new TimeoutError(ms));
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

// ─── 导出 ────────────────────────────────────────────────────────

export { Semaphore, TimeoutError, ToolNotFoundError, hasConflict, buildLayers };
export type { ToolDependency } from '../core/tool.js';
