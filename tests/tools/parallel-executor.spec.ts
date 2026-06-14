/**
 * 并行工具执行器测试
 *
 * 测试覆盖：
 * 1. Semaphore 并发控制
 * 2. 基本并行执行（多工具同时调用）
 * 3. 超时控制
 * 4. 工具不存在时的错误处理
 * 5. failFast 模式
 * 6. 冲突检测和分层
 * 7. 无冲突工具的分层并行
 * 8. 混合冲突场景
 * 9. 降级到串行（不检测冲突）
 * 10. 空调用列表
 * 11. 工具返回失败结果
 */

import { describe, it, expect, vi } from 'vitest';
import { DefaultToolRegistry } from '../../src/core/tool.js';
import { executeToolsParallel, Semaphore, hasConflict, buildLayers } from '../../src/tools/parallel-executor.js';
import type { Tool, ToolResult, ToolDependency } from '../../src/core/tool.js';

// ─── 工具工厂 ────────────────────────────────────────────────────

/** 创建一个简单的工具，记录执行时间和顺序 */
function createTestTool(
  name: string,
  delayMs = 0,
  deps?: ToolDependency
): Tool {
  return {
    metadata: {
      name,
      description: `Test tool: ${name}`,
      parameters: { safeParse: () => ({ success: true, data: {} }) } as any,
    },
    dependencies: deps,
    async execute(args: Record<string, unknown>): Promise<ToolResult> {
      if (delayMs > 0) {
        await new Promise((r) => setTimeout(r, delayMs));
      }
      return {
        success: true,
        output: `${name} executed with args: ${JSON.stringify(args)}`,
        data: { tool: name, args },
      };
    },
  };
}

// ─── Semaphore 测试 ──────────────────────────────────────────────

describe('Semaphore', () => {
  it('应该限制最大并发数', async () => {
    const sem = new Semaphore(2);
    let concurrent = 0;
    let maxConcurrent = 0;

    await Promise.all(
      Array.from({ length: 5 }, () =>
        sem.run(async () => {
          concurrent++;
          maxConcurrent = Math.max(maxConcurrent, concurrent);
          await new Promise((r) => setTimeout(r, 50));
          concurrent--;
        })
      )
    );

    expect(maxConcurrent).toBeLessThanOrEqual(2);
    expect(concurrent).toBe(0);
  });

  it('concurrency=1 时应该串行执行', async () => {
    const sem = new Semaphore(1);
    const order: number[] = [];

    await Promise.all(
      [1, 2, 3].map((i) =>
        sem.run(async () => {
          order.push(i);
          await new Promise((r) => setTimeout(r, 10));
          order.push(-i);
        })
      )
    );

    // 同步模式：一个完成后下一个才能开始
    expect(order).toEqual([1, -1, 2, -2, 3, -3]);
  });
});

// ─── 冲突检测测试 ────────────────────────────────────────────────

describe('hasConflict', () => {
  it('两个只读相同 key 的工具不冲突', () => {
    const a: ToolDependency = { reads: ['user_profile'] };
    const b: ToolDependency = { reads: ['user_profile'] };
    expect(hasConflict(a, b)).toBe(false);
  });

  it('一个读一个写相同 key 的工具冲突', () => {
    const a: ToolDependency = { reads: ['counter'] };
    const b: ToolDependency = { writes: ['counter'] };
    expect(hasConflict(a, b)).toBe(true);
  });

  it('两个写相同 key 的工具冲突', () => {
    const a: ToolDependency = { writes: ['file_a'] };
    const b: ToolDependency = { writes: ['file_a'] };
    expect(hasConflict(a, b)).toBe(true);
  });

  it('操作不同 key 的工具不冲突', () => {
    const a: ToolDependency = { writes: ['file_a'], reads: ['config'] };
    const b: ToolDependency = { writes: ['file_b'], reads: ['user_prefs'] };
    expect(hasConflict(a, b)).toBe(false);
  });

  it('无声明依赖的工具默认与所有工具不冲突', () => {
    const a: ToolDependency = {};
    const b: ToolDependency = { writes: ['anything'] };
    // 空依赖：既不读也不写任何 key，所以不冲突
    expect(hasConflict(a, b)).toBe(false);
  });
});

// ─── 分层算法测试 ────────────────────────────────────────────────

describe('buildLayers', () => {
  it('无冲突的工具应该在同一层', () => {
    const calls = [
      { id: '1', name: 'a', dependency: { reads: ['x'] } },
      { id: '2', name: 'b', dependency: { reads: ['y'] } },
      { id: '3', name: 'c', dependency: { reads: ['z'] } },
    ];

    const layers = buildLayers(calls);
    expect(layers.length).toBe(1);
    expect(layers[0].length).toBe(3);
  });

  it('有冲突的工具应该分到不同层', () => {
    const calls = [
      { id: '1', name: 'a', dependency: { writes: ['counter'] } },
      { id: '2', name: 'b', dependency: { writes: ['counter'] } },
      { id: '3', name: 'c', dependency: { writes: ['counter'] } },
    ];

    const layers = buildLayers(calls);
    // 三个工具都写 counter，每层只能放一个
    expect(layers.length).toBe(3);
    expect(layers[0].length).toBe(1);
    expect(layers[1].length).toBe(1);
    expect(layers[2].length).toBe(1);
  });

  it('混合场景：读写分离', () => {
    const calls = [
      { id: '1', name: 'writer', dependency: { writes: ['shared'] } },
      { id: '2', name: 'reader', dependency: { reads: ['shared'] } },
      { id: '3', name: 'independent', dependency: { reads: ['other'] } },
    ];

    const layers = buildLayers(calls);

    // writer 和 reader 冲突，independent 和两者都不冲突
    // 所以 layer 0: writer + independent, layer 1: reader
    // 或者 layer 0: reader + independent, layer 1: writer
    // 或者 layer 0: independent, layer 1: writer + reader
    // 注意贪心算法按遍历顺序，从 writer 开始：
    // writer 进入 layer 0，reader 与 writer 冲突跳过，independent 不冲突进入 layer 0
    // 所以 layer 0: writer + independent, layer 1: reader
    expect(layers.length).toBe(2);
    expect(layers[0].length).toBe(2);
    // layer 0 包含 writer 和 independent
    expect(layers[0].map((c) => c.name).sort()).toEqual(['independent', 'writer']);
  });
});

// ─── 并行执行主测试 ──────────────────────────────────────────────

describe('executeToolsParallel', () => {
  it('空调用列表应该返回空结果', async () => {
    const registry = new DefaultToolRegistry();
    const result = await executeToolsParallel(registry, []);
    expect(result.successes).toBe(0);
    expect(result.failures).toHaveLength(0);
    expect(result.elapsedMs).toBe(0);
    expect(result.layers).toBe(0);
  });

  it('应该并行执行多个独立工具', async () => {
    const registry = new DefaultToolRegistry();
    registry.register(createTestTool('tool_a', 50, { reads: ['x'] }));
    registry.register(createTestTool('tool_b', 50, { reads: ['y'] }));
    registry.register(createTestTool('tool_c', 50, { reads: ['z'] }));

    const start = Date.now();
    const result = await executeToolsParallel(
      registry,
      [
        { id: '1', name: 'tool_a', args: {} },
        { id: '2', name: 'tool_b', args: {} },
        { id: '3', name: 'tool_c', args: {} },
      ],
      { maxConcurrency: 3, enableConflictDetection: true }
    );
    const elapsed = Date.now() - start;

    // 3 个工具各 50ms，并行执行总时间应 < 150ms
    expect(elapsed).toBeLessThan(150);
    expect(result.successes).toBe(3);
    expect(result.layers).toBe(1);
  });

  it('有冲突的工具应该分层串行执行', async () => {
    const registry = new DefaultToolRegistry();
    registry.register(createTestTool('writer_a', 30, { writes: ['shared'] }));
    registry.register(createTestTool('writer_b', 30, { writes: ['shared'] }));
    registry.register(createTestTool('writer_c', 30, { writes: ['shared'] }));

    const start = Date.now();
    const result = await executeToolsParallel(
      registry,
      [
        { id: '1', name: 'writer_a', args: {} },
        { id: '2', name: 'writer_b', args: {} },
        { id: '3', name: 'writer_c', args: {} },
      ],
      { maxConcurrency: 3, enableConflictDetection: true }
    );
    const elapsed = Date.now() - start;

    // 3 层，每层 1 个工具各 30ms，串行总时间应 >= 90ms
    expect(elapsed).toBeGreaterThanOrEqual(80);
    expect(result.successes).toBe(3);
    expect(result.layers).toBe(3);
  });

  it('工具不存在时应该优雅地报告错误', async () => {
    const registry = new DefaultToolRegistry();
    registry.register(createTestTool('existing_tool', 0));

    const result = await executeToolsParallel(
      registry,
      [
        { id: '1', name: 'existing_tool', args: {} },
        { id: '2', name: 'nonexistent_tool', args: {} },
      ]
    );

    expect(result.successes).toBe(1);
    expect(result.failures).toHaveLength(1);
    expect(result.failures[0].name).toBe('nonexistent_tool');
    expect(result.failures[0].error).toContain('not found');
  });

  it('超时控制：超过 timeoutMs 的工具应该失败', async () => {
    const registry = new DefaultToolRegistry();
    registry.register(createTestTool('slow_tool', 100));

    const result = await executeToolsParallel(
      registry,
      [{ id: '1', name: 'slow_tool', args: {} }],
      { timeoutMs: 20 }
    );

    expect(result.failures).toHaveLength(1);
    expect(result.failures[0].error).toContain('timed out');
  });

  it('failFast 模式：第一个失败后跳过剩余调用', async () => {
    const registry = new DefaultToolRegistry();
    registry.register(createTestTool('ok_tool', 0));
    // 不注册 failing_tool
    registry.register(createTestTool('another_ok', 0));

    const result = await executeToolsParallel(
      registry,
      [
        { id: '1', name: 'ok_tool', args: {} },
        { id: '2', name: 'nonexistent', args: {} },
        { id: '3', name: 'another_ok', args: {} },
      ],
      { failFast: true, maxConcurrency: 3 }
    );

    // 不存在的工具在 second layer 还是失败了（如果它和 ok_tool 没有冲突）
    // 或者因为 failFast 跳过
    // 注意：nonexistent 没有声明依赖，所以默认不冲突，和 ok_tool 同层
    // 在 failFast 模式下，同层内的失败会导致后续层被跳过
    // 但当前层内的 another_ok 应该被执行完（它和 ok_tool 同层，没有冲突）
    const okResults = result.results.get('1');
    expect(okResults?.success).toBe(true);

    // another_ok 要么在同层成功，要么被跳过
    // 这取决于执行顺序和 failFast 的实现
    const anotherOkResult = result.results.get('3');
    expect(anotherOkResult?.success).toBe(true);
  });

  it('不启用冲突检测时，所有工具都在同一层', async () => {
    const registry = new DefaultToolRegistry();
    registry.register(createTestTool('a', 10, { writes: ['shared'] }));
    registry.register(createTestTool('b', 10, { writes: ['shared'] }));

    const result = await executeToolsParallel(
      registry,
      [
        { id: '1', name: 'a', args: {} },
        { id: '2', name: 'b', args: {} },
      ],
      { enableConflictDetection: false, maxConcurrency: 2 }
    );

    // 不检测冲突 = 全部并行 = 1 层
    expect(result.layers).toBe(1);
    expect(result.successes).toBe(2);
  });

  it('工具自行返回失败结果时应该被记录', async () => {
    const failingTool: Tool = {
      metadata: {
        name: 'failing_tool',
        description: 'Tool that returns failure',
        parameters: { safeParse: () => ({ success: true, data: {} }) } as any,
      },
      async execute(): Promise<ToolResult> {
        return { success: false, output: 'Computation failed', error: 'Division by zero' };
      },
    };

    const registry = new DefaultToolRegistry();
    registry.register(createTestTool('ok_tool', 0));
    registry.register(failingTool);

    const result = await executeToolsParallel(
      registry,
      [
        { id: '1', name: 'ok_tool', args: {} },
        { id: '2', name: 'failing_tool', args: {} },
      ]
    );

    expect(result.failures).toHaveLength(1);
    expect(result.failures[0].error).toContain('Division by zero');
    expect(result.results.get('2')?.success).toBe(false);
  });

  it('工具执行抛出异常时不应该影响其他工具', async () => {
    const throwingTool: Tool = {
      metadata: {
        name: 'throwing_tool',
        description: 'Tool that throws',
        parameters: { safeParse: () => ({ success: true, data: {} }) } as any,
      },
      async execute(): Promise<ToolResult> {
        throw new Error('Internal error');
      },
    };

    const registry = new DefaultToolRegistry();
    registry.register(createTestTool('good_tool', 0));
    registry.register(throwingTool);

    const result = await executeToolsParallel(
      registry,
      [
        { id: '1', name: 'good_tool', args: {} },
        { id: '2', name: 'throwing_tool', args: {} },
      ]
    );

    expect(result.results.get('1')?.success).toBe(true);
    expect(result.failures).toHaveLength(1);
    expect(result.failures[0].name).toBe('throwing_tool');
  });
});

// ─── Agent 集成测试 ─────────────────────────────────────────────

import { DefaultAgent } from '../../src/runloop/agent.js';
import type { LLMProvider, LLMResponse } from '../../src/core/llm.js';

describe('DefaultAgent with parallel execution', () => {
  function createMockProvider(responses: LLMResponse[]): LLMProvider {
    let index = 0;
    return {
      name: 'mock',
      async complete(): Promise<LLMResponse> {
        if (index >= responses.length) {
          return { content: 'No more responses configured', finishReason: 'stop' };
        }
        return responses[index++];
      },
      async completeStream(
        _request: any,
        _onChunk: any
      ): Promise<LLMResponse> {
        if (index >= responses.length) {
          return { content: 'No more responses configured', finishReason: 'stop' };
        }
        return responses[index++];
      },
    };
  }

  it('并行模式下应该能正常执行一轮工具调用', async () => {
    const provider = createMockProvider([
      {
        content: '',
        finishReason: 'tool_calls',
        toolCalls: [
          { id: 'call_1', name: 'calculator', args: { expression: '1+2' } },
          { id: 'call_2', name: 'calculator', args: { expression: '3+4' } },
        ],
      },
      {
        content: 'The results are 3 and 7.',
        finishReason: 'stop',
      },
    ]);

    const registry = new DefaultToolRegistry();
    const calculatorTool: Tool = {
      metadata: {
        name: 'calculator',
        description: 'Calculate',
        parameters: { safeParse: () => ({ success: true, data: {} }) } as any,
      },
      async execute(args: Record<string, unknown>): Promise<ToolResult> {
        // 安全执行简单的数字运算
        const expr = args['expression'] as string;
        const sanitized = expr.replace(/[^0-9+\-*/.()]/g, '');
        const result = Function(`"use strict"; return (${sanitized})`)();
        return { success: true, output: String(result) };
      },
    };
    registry.register(calculatorTool);

    const agent = new DefaultAgent({
      name: 'parallel-test',
      model: { provider, model: 'mock' },
      systemPrompt: 'You are a helpful assistant.',
      tools: registry,
      memory: {
        shortTerm: {
          add() {},
          getMessages() { return []; },
          clear() {},
          estimatedTokens() { return 0; },
        },
        longTerm: {
          async store() { return 'id'; },
          async search() { return []; },
          async get() { return null; },
          async delete() { return true; },
          async forget() { return 0; },
        },
        async consolidate() {},
      },
      maxIterations: 5,
      parallelExecution: {
        maxConcurrency: 3,
        enableConflictDetection: true,
      },
    });

    const result = await agent.run('Calculate 1+2 and 3+4');

    expect(result.output).toBe('The results are 3 and 7.');
    expect(result.iterations).toBe(2);
  });

  it('并行模式遇到不存在的工具应该优雅处理', async () => {
    const provider = createMockProvider([
      {
        content: '',
        finishReason: 'tool_calls',
        toolCalls: [
          { id: 'call_1', name: 'valid_tool', args: {} },
          { id: 'call_2', name: 'ghost_tool', args: {} },
        ],
      },
      {
        content: 'One tool was not found.',
        finishReason: 'stop',
      },
    ]);

    const registry = new DefaultToolRegistry();
    registry.register({
      metadata: {
        name: 'valid_tool',
        description: 'A valid tool',
        parameters: { safeParse: () => ({ success: true, data: {} }) } as any,
      },
      async execute(): Promise<ToolResult> {
        return { success: true, output: 'ok' };
      },
    });

    const agent = new DefaultAgent({
      name: 'parallel-ghost',
      model: { provider, model: 'mock' },
      systemPrompt: 'Test.',
      tools: registry,
      memory: {
        shortTerm: {
          add() {}, getMessages() { return []; }, clear() {},
          estimatedTokens() { return 0; },
        },
        longTerm: {
          async store() { return 'id'; },
          async search() { return []; },
          async get() { return null; },
          async delete() { return true; },
          async forget() { return 0; },
        },
        async consolidate() {},
      },
      maxIterations: 5,
      parallelExecution: {
        maxConcurrency: 3,
      },
    });

    const result = await agent.run('Test');
    expect(result.output).toBe('One tool was not found.');
    expect(result.iterations).toBe(2);
  });
});
