/**
 * Tool → Skill 适配器测试
 *
 * 覆盖：
 * - toSkill() 创建 Skill
 * - groupToolsByPrefix() 自动分组
 * - ToolRegistryAdapter 合并逻辑
 */

import { describe, it, expect } from 'vitest';
import { z } from 'zod';
import type { Tool, ToolResult } from '../../core/tool.js';
import {
  toSkill,
  groupToolsByPrefix,
  ToolRegistryAdapter,
} from '../tool-adapter.js';
import type { Skill } from '../skill.js';

// ─── 测试用 Tool ─────────────────────────────────────────────────

const searchTool: Tool = {
  metadata: {
    name: 'web_search',
    description: 'Search the web',
    parameters: z.object({ query: z.string() }),
  },
  async execute(args: Record<string, unknown>): Promise<ToolResult> {
    return { success: true, output: `Search results for: ${args['query']}` };
  },
};

const fetchTool: Tool = {
  metadata: {
    name: 'web_fetch',
    description: 'Fetch a URL',
    parameters: z.object({ url: z.string() }),
  },
  async execute(args: Record<string, unknown>): Promise<ToolResult> {
    return { success: true, output: `Content from: ${args['url']}` };
  },
};

const calculatorTool: Tool = {
  metadata: {
    name: 'math_calc',
    description: 'Do calculation',
    parameters: z.object({ expr: z.string() }),
  },
  async execute(args: Record<string, unknown>): Promise<ToolResult> {
    return { success: true, output: `Result: ${args['expr']}` };
  },
};

const statsTool: Tool = {
  metadata: {
    name: 'math_stats',
    description: 'Compute statistics',
    parameters: z.object({ numbers: z.array(z.number()) }),
  },
  async execute(args: Record<string, unknown>): Promise<ToolResult> {
    const nums = args['numbers'] as number[];
    return { success: true, output: `Stats for ${nums.length} numbers` };
  },
};

// ─── toSkill ──────────────────────────────────────────────────────

describe('toSkill()', () => {
  it('creates a Skill from tools', () => {
    const skill = toSkill({
      name: 'web-tools',
      description: 'Web browsing tools',
      tags: ['web', 'search'],
      tools: [searchTool, fetchTool],
    });

    expect(skill.metadata.name).toBe('web-tools');
    expect(skill.metadata.toolNames).toEqual(['web_search', 'web_fetch']);
    expect(skill.metadata.version).toBe('1.0.0');
    expect(skill.metadata.tags).toEqual(['web', 'search']);
  });

  it('returns tools via getTools()', () => {
    const skill = toSkill({
      name: 'web-tools',
      description: 'Web tools',
      tools: [searchTool],
    });

    const tools = skill.getTools();
    expect(tools).toHaveLength(1);
    expect(tools[0].metadata.name).toBe('web_search');
  });

  it('accepts custom version', () => {
    const skill = toSkill({
      name: 'web-tools',
      description: 'Web tools',
      version: '2.0.0',
      tools: [searchTool],
    });

    expect(skill.metadata.version).toBe('2.0.0');
  });

  it('includes init and dispose when provided', () => {
    let initialized = false;
    const skill = toSkill({
      name: 'test',
      description: 'Test',
      tools: [searchTool],
      init: async () => { initialized = true; },
      dispose: async () => { initialized = false; },
    });

    expect(skill.init).toBeDefined();
    expect(skill.dispose).toBeDefined();
  });

  it('omits init and dispose when not provided', () => {
    const skill = toSkill({
      name: 'test',
      description: 'Test',
      tools: [searchTool],
    });

    expect(skill.init).toBeUndefined();
    expect(skill.dispose).toBeUndefined();
  });

  it('creates a Skill that satisfies the Skill interface', () => {
    const skill: Skill = toSkill({
      name: 'web-tools',
      description: 'Web tools',
      tools: [searchTool],
    });

    expect(skill.metadata.name).toBe('web-tools');
    expect(skill.metadata.description).toBe('Web tools');
  });

  it('uses empty array for tags when not specified', () => {
    const skill = toSkill({
      name: 'test',
      description: 'Test',
      tools: [searchTool],
    });

    expect(skill.metadata.tags).toEqual([]);
  });
});

// ─── groupToolsByPrefix ──────────────────────────────────────────

describe('groupToolsByPrefix()', () => {
  it('groups tools by name prefix', () => {
    const allTools = [searchTool, fetchTool, calculatorTool, statsTool];
    const skills = groupToolsByPrefix(allTools);

    expect(skills).toHaveLength(2);
    const names = skills.map((s) => s.metadata.name).sort();
    expect(names).toEqual(['math-tools', 'web-tools']);
  });

  it('each skill contains the correct tools', () => {
    const allTools = [searchTool, fetchTool, calculatorTool];
    const skills = groupToolsByPrefix(allTools);

    const webSkill = skills.find((s) => s.metadata.name === 'web-tools')!;
    expect(webSkill.getTools().map((t) => t.metadata.name).sort()).toEqual([
      'web_fetch',
      'web_search',
    ]);

    const mathSkill = skills.find((s) => s.metadata.name === 'math-tools')!;
    expect(mathSkill.getTools().map((t) => t.metadata.name)).toEqual([
      'math_calc',
    ]);
  });

  it('accepts custom descriptions', () => {
    const skills = groupToolsByPrefix([searchTool, fetchTool], {
      web: 'Custom web description',
    });

    const webSkill = skills.find((s) => s.metadata.name === 'web-tools')!;
    expect(webSkill.metadata.description).toBe('Custom web description');
  });

  it('returns empty array for empty input', () => {
    const skills = groupToolsByPrefix([]);
    expect(skills).toHaveLength(0);
  });

  it('handles tools with same prefix prefix', () => {
    const tool1: Tool = {
      metadata: {
        name: 'db_query',
        description: 'Query DB',
        parameters: z.object({}),
      },
      async execute() {
        return { success: true, output: 'ok' };
      },
    };
    const tool2: Tool = {
      metadata: {
        name: 'db_insert',
        description: 'Insert to DB',
        parameters: z.object({}),
      },
      async execute() {
        return { success: true, output: 'ok' };
      },
    };

    const skills = groupToolsByPrefix([tool1, tool2]);
    expect(skills).toHaveLength(1);
    expect(skills[0].metadata.name).toBe('db-tools');
    expect(skills[0].getTools()).toHaveLength(2);
  });
});

// ─── ToolRegistryAdapter ─────────────────────────────────────────

describe('ToolRegistryAdapter', () => {
  it('starts with no global tools', () => {
    const adapter = new ToolRegistryAdapter();
    expect(adapter.getGlobalNames()).toHaveLength(0);
  });

  it('registers global tools', () => {
    const adapter = new ToolRegistryAdapter();
    adapter.registerGlobal(searchTool);

    expect(adapter.getGlobalNames()).toEqual(['web_search']);
  });

  it('registers multiple global tools', () => {
    const adapter = new ToolRegistryAdapter();
    adapter.registerGlobalMany([searchTool, fetchTool]);

    expect(adapter.getGlobalNames()).toEqual(['web_search', 'web_fetch']);
  });

  describe('merge()', () => {
    it('returns only global tools when no skill tools active', () => {
      const adapter = new ToolRegistryAdapter();
      adapter.registerGlobal(searchTool);

      const { metadata, toolMap } = adapter.merge([]);

      expect(metadata).toHaveLength(1);
      expect(metadata[0].name).toBe('web_search');
      expect(toolMap.has('web_search')).toBe(true);
    });

    it('merges skill tools with global tools', () => {
      const adapter = new ToolRegistryAdapter();
      adapter.registerGlobal(searchTool);
      adapter.registerGlobal(calculatorTool);

      const { metadata, toolMap } = adapter.merge([fetchTool]);

      expect(metadata).toHaveLength(3);
      expect(toolMap.has('web_search')).toBe(true);
      expect(toolMap.has('math_calc')).toBe(true);
      expect(toolMap.has('web_fetch')).toBe(true);
    });

    it('skill tools override global tools with same name', () => {
      const adapter = new ToolRegistryAdapter();
      const globalVersion: Tool = {
        metadata: {
          name: 'web_search',
          description: 'Global search',
          parameters: z.object({ q: z.string() }),
        },
        async execute() {
          return { success: true, output: 'global' };
        },
      };
      const skillVersion: Tool = {
        metadata: {
          name: 'web_search',
          description: 'Skill search',
          parameters: z.object({ q: z.string() }),
        },
        async execute() {
          return { success: true, output: 'skill' };
        },
      };

      adapter.registerGlobal(globalVersion);
      const { metadata, toolMap } = adapter.merge([skillVersion]);

      // Only one web_search in result (skill version wins)
      const searchMetadata = metadata.filter((m) => m.name === 'web_search');
      expect(searchMetadata).toHaveLength(1);
      expect(searchMetadata[0].description).toBe('Skill search');

      // toolMap should return skill version
      expect(toolMap.get('web_search')).toBe(skillVersion);
    });

    it('returns empty when no tools registered at all', () => {
      const adapter = new ToolRegistryAdapter();
      const { metadata, toolMap } = adapter.merge([]);
      expect(metadata).toHaveLength(0);
      expect(toolMap.size).toBe(0);
    });

    it('is idempotent when merging same data multiple times', () => {
      const adapter = new ToolRegistryAdapter();
      adapter.registerGlobal(searchTool);

      const result1 = adapter.merge([fetchTool]);
      const result2 = adapter.merge([fetchTool]);

      expect(result1.metadata).toEqual(result2.metadata);
    });
  });

  describe('markOverridden()', () => {
    it('marks a tool as overridden by skill', () => {
      const adapter = new ToolRegistryAdapter();
      adapter.registerGlobal(searchTool);
      adapter.markOverridden('web_search');

      expect(adapter.getGlobalNames()).toContain('web_search');
    });
  });
});
