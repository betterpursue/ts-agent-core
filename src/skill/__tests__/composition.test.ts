/**
 * Skill 组合与编排测试
 *
 * 覆盖：
 * - composedSkill() 创建组合 Skill
 * - SkillChain 执行链
 * - validateChainDependencies 验证依赖
 */

import { describe, it, expect, vi } from 'vitest';
import { z } from 'zod';
import type { Tool, ToolResult } from '../../core/tool.js';
import {
  composedSkill,
  SkillChain,
  validateChainDependencies,
} from '../composition.js';
import type { Skill, SkillDependency } from '../composition.js';

// ─── 测试用 Tool ─────────────────────────────────────────────────

const searchTool: Tool = {
  metadata: {
    name: 'search',
    description: 'Search the web',
    parameters: z.object({ query: z.string() }),
  },
  async execute(args: Record<string, unknown>): Promise<ToolResult> {
    return { success: true, output: `Search: ${args['query']}` };
  },
};

const summarizeTool: Tool = {
  metadata: {
    name: 'summarize',
    description: 'Summarize text',
    parameters: z.object({ text: z.string() }),
  },
  async execute(args: Record<string, unknown>): Promise<ToolResult> {
    return { success: true, output: `Summary: ${String(args['text']).slice(0, 20)}...` };
  },
};

const formatTool: Tool = {
  metadata: {
    name: 'format',
    description: 'Format text',
    parameters: z.object({ text: z.string() }),
  },
  async execute(args: Record<string, unknown>): Promise<ToolResult> {
    return { success: true, output: `Formatted: ${args['text']}` };
  },
};

// ─── 辅助函数 ────────────────────────────────────────────────────

function createMockSkill(
  name: string,
  tools: Tool[],
  initImpl?: () => Promise<void>,
  disposeImpl?: () => Promise<void>
): Skill {
  return {
    metadata: {
      name,
      description: `${name} skill`,
      version: '1.0.0',
      tags: [name],
      toolNames: tools.map((t) => t.metadata.name),
    },
    getTools: () => tools,
    ...(initImpl ? { init: initImpl } : {}),
    ...(disposeImpl ? { dispose: disposeImpl } : {}),
  };
}

// ─── composedSkill ──────────────────────────────────────────────

describe('composedSkill()', () => {
  it('creates a Skill from multiple sub-skills', () => {
    const searchSkill = createMockSkill('search', [searchTool]);
    const summarySkill = createMockSkill('summarize', [summarizeTool]);

    const composed = composedSkill({
      name: 'research',
      description: 'Research workflow',
      tags: ['research'],
      skills: [searchSkill, summarySkill],
    });

    expect(composed.metadata.name).toBe('research');
    expect(composed.metadata.description).toBe('Research workflow');
    expect(composed.metadata.toolNames).toEqual(['search', 'summarize']);
  });

  it('aggregates tools from all sub-skills', () => {
    const searchSkill = createMockSkill('search', [searchTool]);
    const summarySkill = createMockSkill('summarize', [summarizeTool]);

    const composed = composedSkill({
      name: 'research',
      description: 'Research',
      skills: [searchSkill, summarySkill],
    });

    const tools = composed.getTools();
    expect(tools).toHaveLength(2);
    const names = tools.map((t) => t.metadata.name).sort();
    expect(names).toEqual(['search', 'summarize']);
  });

  it('removes duplicate tools across sub-skills', () => {
    const searchSkill = createMockSkill('search', [searchTool]);
    const anotherSearch = createMockSkill('search2', [searchTool]); // 同名工具

    const composed = composedSkill({
      name: 'multi-search',
      description: 'Multiple search',
      skills: [searchSkill, anotherSearch],
    });

    const tools = composed.getTools();
    expect(tools).toHaveLength(1);
    expect(tools[0].metadata.name).toBe('search');
  });

  it('calls init on all sub-skills when activated', async () => {
    const initFn1 = vi.fn().mockResolvedValue(undefined);
    const initFn2 = vi.fn().mockResolvedValue(undefined);

    const skill1 = createMockSkill('s1', [searchTool], initFn1);
    const skill2 = createMockSkill('s2', [summarizeTool], initFn2);

    const composed = composedSkill({
      name: 'combined',
      description: 'Combined',
      skills: [skill1, skill2],
    });

    await composed.init?.();

    expect(initFn1).toHaveBeenCalledOnce();
    expect(initFn2).toHaveBeenCalledOnce();
  });

  it('calls dispose in reverse order', async () => {
    const disposeOrder: string[] = [];
    const skill1 = createMockSkill('s1', [searchTool], undefined, async () => {
      disposeOrder.push('s1');
    });
    const skill2 = createMockSkill('s2', [summarizeTool], undefined, async () => {
      disposeOrder.push('s2');
    });

    const composed = composedSkill({
      name: 'combined',
      description: 'Combined',
      skills: [skill1, skill2],
    });

    await composed.dispose?.();

    expect(disposeOrder).toEqual(['s2', 's1']);
  });

  it('uses custom toolAggregator when provided', () => {
    const searchSkill = createMockSkill('search', [searchTool, formatTool]);

    const composed = composedSkill({
      name: 'custom',
      description: 'Custom aggregator',
      skills: [searchSkill],
      toolAggregator: (toolLists) => {
        // 只返回第一个 Skill 的工具
        return toolLists[0] ?? [];
      },
    });

    const tools = composed.getTools();
    expect(tools).toHaveLength(2);
  });

  it('uses default version when not provided', () => {
    const skill = createMockSkill('s1', [searchTool]);
    const composed = composedSkill({
      name: 'test',
      description: 'Test',
      skills: [skill],
    });

    expect(composed.metadata.version).toBe('1.0.0');
  });
});

// ─── SkillChain ──────────────────────────────────────────────────

describe('SkillChain', () => {
  it('executes skills in order', async () => {
    const order: string[] = [];
    const s1 = createMockSkill('s1', [searchTool], async () => order.push('init-s1'));
    const s2 = createMockSkill('s2', [summarizeTool], async () => order.push('init-s2'));

    const chain = new SkillChain({
      skills: [s1, s2],
    });

    const results = await chain.execute();

    expect(order).toEqual(['init-s1', 'init-s2']);
    expect(results).toHaveLength(2);
    expect(results[0]).toHaveLength(1);
    expect(results[0][0].metadata.name).toBe('search');
    expect(results[1][0].metadata.name).toBe('summarize');
  });

  it('getAllTools returns merged tool list', () => {
    const s1 = createMockSkill('s1', [searchTool]);
    const s2 = createMockSkill('s2', [summarizeTool]);

    const chain = new SkillChain({
      skills: [s1, s2],
    });

    const all = chain.getAllTools();
    expect(all).toHaveLength(2);
    const names = all.map((t) => t.metadata.name).sort();
    expect(names).toEqual(['search', 'summarize']);
  });

  it('getSummaries returns all skill metadata', () => {
    const s1 = createMockSkill('s1', [searchTool]);
    s1.metadata.version = 'v2';
    const s2 = createMockSkill('s2', [summarizeTool]);

    const chain = new SkillChain({
      skills: [s1, s2],
    });

    const summaries = chain.getSummaries();
    expect(summaries).toHaveLength(2);
    expect(summaries[0].name).toBe('s1');
    expect(summaries[0].version).toBe('v2');
  });

  it('calls afterEach hook', async () => {
    const afterEach = vi.fn();
    const s1 = createMockSkill('s1', [searchTool]);
    const s2 = createMockSkill('s2', [summarizeTool]);

    const chain = new SkillChain({
      skills: [s1, s2],
      afterEach,
    });

    await chain.execute();

    expect(afterEach).toHaveBeenCalledTimes(2);
    expect(afterEach).toHaveBeenNthCalledWith(1, s1, [searchTool]);
    expect(afterEach).toHaveBeenNthCalledWith(2, s2, [summarizeTool]);
  });

  it('calls afterAll hook with all results', async () => {
    const afterAll = vi.fn();
    const s1 = createMockSkill('s1', [searchTool]);
    const s2 = createMockSkill('s2', [summarizeTool]);

    const chain = new SkillChain({
      skills: [s1, s2],
      afterAll,
    });

    await chain.execute();

    expect(afterAll).toHaveBeenCalledOnce();
    expect(afterAll).toHaveBeenCalledWith([
      [searchTool],
      [summarizeTool],
    ]);
  });

  it('handles empty chain', async () => {
    const chain = new SkillChain({
      skills: [],
    });

    const results = await chain.execute();
    expect(results).toHaveLength(0);
    expect(chain.getAllTools()).toHaveLength(0);
    expect(chain.getSummaries()).toHaveLength(0);
  });
});

// ─── validateChainDependencies ──────────────────────────────────

describe('validateChainDependencies()', () => {
  it('returns valid when all required dependencies are present', () => {
    const s1 = createMockSkill('search', [searchTool]);
    const s2 = createMockSkill('summarize', [summarizeTool]);

    const result = validateChainDependencies([s1, s2], [
      { name: 'search', type: 'required' },
    ]);

    expect(result.valid).toBe(true);
    expect(result.missing).toHaveLength(0);
  });

  it('returns invalid when required dependency is missing', () => {
    const s1 = createMockSkill('search', [searchTool]);

    const result = validateChainDependencies([s1], [
      { name: 'missing', type: 'required' },
    ]);

    expect(result.valid).toBe(false);
    expect(result.missing).toEqual(['missing']);
  });

  it('ignores optional missing dependencies', () => {
    const s1 = createMockSkill('search', [searchTool]);

    const result = validateChainDependencies([s1], [
      { name: 'missing', type: 'optional' },
    ]);

    expect(result.valid).toBe(true);
    expect(result.missing).toHaveLength(0);
  });

  it('returns empty missing for empty dependencies', () => {
    const s1 = createMockSkill('search', [searchTool]);

    const result = validateChainDependencies([s1], []);

    expect(result.valid).toBe(true);
    expect(result.missing).toHaveLength(0);
  });
});
