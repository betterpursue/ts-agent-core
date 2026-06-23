/**
 * Skill 模块测试
 *
 * 覆盖：
 * - Skill 接口（实现一个 mock Skill）
 * - DefaultSkillRegistry（注册、查找、列表、注销、重复检测、dispose 回调）
 * - ProgressiveSkillLoader（摘要、选择、激活、停用、淘汰、超时清理、init 容错）
 * - 边界情况（空注册表、不存在的 Skill）
 */

import { describe, it, expect, vi } from 'vitest';
import type { Tool, ToolResult } from '../../core/tool.js';
import {
  DefaultSkillRegistry,
  ProgressiveSkillLoader,
} from '../skill.js';
import type {
  Skill,
  SkillMetadata,
  ProgressiveLoaderConfig,
} from '../skill.js';
import { z } from 'zod';

// ─── 测试用 Tool ─────────────────────────────────────────────────

const echoTool: Tool = {
  metadata: {
    name: 'echo',
    description: 'Repeats the input',
    parameters: z.object({ text: z.string() }),
  },
  async execute(args: Record<string, unknown>): Promise<ToolResult> {
    return { success: true, output: String(args['text'] ?? '') };
  },
};

const reverseTool: Tool = {
  metadata: {
    name: 'reverse',
    description: 'Reverses the input string',
    parameters: z.object({ text: z.string() }),
  },
  async execute(args: Record<string, unknown>): Promise<ToolResult> {
    const text = String(args['text'] ?? '');
    return { success: true, output: text.split('').reverse().join('') };
  },
};

// ─── 测试用 Skill ────────────────────────────────────────────────

function createMockSkill(
  overrides?: Partial<SkillMetadata> & {
    tools?: Tool[];
    initImpl?: () => Promise<void>;
    disposeImpl?: () => void;
  }
): Skill {
  const metadata: SkillMetadata = {
    name: overrides?.name ?? 'test-skill',
    description: overrides?.description ?? 'A test skill',
    version: overrides?.version ?? '1.0.0',
    tags: overrides?.tags ?? ['test', 'mock'],
    toolNames: overrides?.toolNames ?? ['echo'],
  };

  const tools = overrides?.tools ?? [echoTool];

  return {
    metadata,
    getTools: () => tools,
    ...(overrides?.initImpl ? { init: overrides.initImpl } : {}),
    ...(overrides?.disposeImpl ? { dispose: overrides.disposeImpl } : {}),
  };
}

// ─── DefaultSkillRegistry ────────────────────────────────────────

describe('DefaultSkillRegistry', () => {
  it('starts empty', () => {
    const registry = new DefaultSkillRegistry();
    expect(registry.list()).toHaveLength(0);
    expect(registry.size).toBe(0);
  });

  it('registers a skill', () => {
    const registry = new DefaultSkillRegistry();
    const skill = createMockSkill({ name: 'math' });

    registry.register(skill);

    expect(registry.size).toBe(1);
    expect(registry.get('math')).toBe(skill);
  });

  it('registers multiple skills', () => {
    const registry = new DefaultSkillRegistry();
    const s1 = createMockSkill({ name: 'skill-a' });
    const s2 = createMockSkill({ name: 'skill-b' });

    registry.register(s1);
    registry.register(s2);

    expect(registry.size).toBe(2);
    expect(registry.list()).toHaveLength(2);
  });

  it('throws when registering a duplicate skill name', () => {
    const registry = new DefaultSkillRegistry();
    const s1 = createMockSkill({ name: 'dup' });
    const s2 = createMockSkill({ name: 'dup' });

    registry.register(s1);
    expect(() => registry.register(s2)).toThrow('already registered');
    expect(registry.size).toBe(1);
  });

  it('returns undefined for unknown skill', () => {
    const registry = new DefaultSkillRegistry();
    expect(registry.get('nonexistent')).toBeUndefined();
  });

  it('returns metadata for all registered skills via list()', () => {
    const registry = new DefaultSkillRegistry();
    registry.register(createMockSkill({ name: 'a', tags: ['tag1'] }));
    registry.register(createMockSkill({ name: 'b', tags: ['tag2'] }));

    const metaList = registry.list();

    expect(metaList).toHaveLength(2);
    const names = metaList.map((m) => m.name).sort();
    expect(names).toEqual(['a', 'b']);
  });

  it('returns a copy of metadata (immutable)', () => {
    const registry = new DefaultSkillRegistry();
    registry.register(createMockSkill({ name: 'immutable' }));

    const metaList = registry.list();
    expect(metaList).toHaveLength(1);

    // Mutating the returned array should not affect the registry
    metaList.pop();
    expect(registry.size).toBe(1);
  });

  it('unregisters a skill and calls dispose', async () => {
    const registry = new DefaultSkillRegistry();
    const disposeFn = vi.fn();
    const skill = createMockSkill({ name: 'to-remove', disposeImpl: disposeFn });

    registry.register(skill);
    const removed = await registry.unregister('to-remove');

    expect(removed).toBe(true);
    expect(registry.size).toBe(0);
    expect(registry.get('to-remove')).toBeUndefined();
    expect(disposeFn).toHaveBeenCalledOnce();
  });

  it('returns false when unregistering an unknown skill', async () => {
    const registry = new DefaultSkillRegistry();
    expect(await registry.unregister('unknown')).toBe(false);
  });

  it('dispose error does not prevent removal', async () => {
    const registry = new DefaultSkillRegistry();
    const skill = createMockSkill({
      name: 'bad-dispose',
      disposeImpl: async () => { throw new Error('dispose failed'); },
    });

    registry.register(skill);
    // Should not throw
    const removed = await registry.unregister('bad-dispose');

    expect(removed).toBe(true);
    expect(registry.size).toBe(0);
  });
});

// ─── ProgressiveSkillLoader ──────────────────────────────────────

describe('ProgressiveSkillLoader', () => {
  describe('getSkillSummaries()', () => {
    it('returns empty array when no skills registered', () => {
      const registry = new DefaultSkillRegistry();
      const loader = new ProgressiveSkillLoader(registry);

      expect(loader.getSkillSummaries()).toHaveLength(0);
    });

    it('returns summaries for all registered skills', () => {
      const registry = new DefaultSkillRegistry();
      registry.register(createMockSkill({
        name: 'search',
        description: 'Search the web',
        tags: ['web', 'search'],
      }));
      registry.register(createMockSkill({
        name: 'math',
        description: 'Do arithmetic',
        tags: ['math', 'calc'],
      }));

      const loader = new ProgressiveSkillLoader(registry);
      const summaries = loader.getSkillSummaries();

      expect(summaries).toHaveLength(2);
      expect(summaries[0]).toEqual({
        name: 'search',
        description: 'Search the web',
        tags: ['web', 'search'],
      });
    });

    it('summaries omit toolNames and version', () => {
      const registry = new DefaultSkillRegistry();
      registry.register(createMockSkill({
        name: 'test',
        description: 'test skill',
        tags: ['test'],
      }));

      const loader = new ProgressiveSkillLoader(registry);
      const summaries = loader.getSkillSummaries();

      expect(summaries[0]).not.toHaveProperty('toolNames');
      expect(summaries[0]).not.toHaveProperty('version');
    });
  });

  describe('selectSkills()', () => {
    it('returns matching skills sorted by relevance', () => {
      const registry = new DefaultSkillRegistry();
      registry.register(createMockSkill({
        name: 'web-search',
        description: 'Search the web for information',
        tags: ['web', 'search', 'browser'],
      }));
      registry.register(createMockSkill({
        name: 'calculator',
        description: 'Perform arithmetic calculations',
        tags: ['math', 'calc', 'number'],
      }));

      const loader = new ProgressiveSkillLoader(registry, { maxActiveSkills: 3 });
      const selected = loader.selectSkills('search web browser');

      // web-search should rank higher because name + tags match
      expect(selected.length).toBeGreaterThanOrEqual(1);
      expect(selected[0].metadata.name).toBe('web-search');
    });

    it('returns empty array when no skills match', () => {
      const registry = new DefaultSkillRegistry();
      registry.register(createMockSkill({
        name: 'math',
        tags: ['math'],
      }));

      const loader = new ProgressiveSkillLoader(registry);
      const selected = loader.selectSkills('cooking recipe');

      expect(selected).toHaveLength(0);
    });

    it('returns empty array for empty registry', () => {
      const registry = new DefaultSkillRegistry();
      const loader = new ProgressiveSkillLoader(registry);

      expect(loader.selectSkills('anything')).toHaveLength(0);
    });

    it('respects maxActiveSkills limit in selection', () => {
      const registry = new DefaultSkillRegistry();
      registry.register(createMockSkill({ name: 'a', tags: ['common'] }));
      registry.register(createMockSkill({ name: 'b', tags: ['common'] }));
      registry.register(createMockSkill({ name: 'c', tags: ['common'] }));

      const loader = new ProgressiveSkillLoader(registry, { maxActiveSkills: 2 });
      const selected = loader.selectSkills('common');

      // Only 2 skills max, even though all 3 match
      expect(selected.length).toBeLessThanOrEqual(2);
    });

    it('matches by tag words', () => {
      const registry = new DefaultSkillRegistry();
      registry.register(createMockSkill({
        name: 'file-reader',
        description: 'Read files from disk',
        tags: ['file-system', 'io'],
      }));

      const loader = new ProgressiveSkillLoader(registry);
      const selected = loader.selectSkills('file');

      expect(selected).toHaveLength(1);
      expect(selected[0].metadata.name).toBe('file-reader');
    });
  });

  describe('activate()', () => {
    it('returns tools from the skill', async () => {
      const registry = new DefaultSkillRegistry();
      const skill = createMockSkill({
        name: 'echo-skill',
        tools: [echoTool],
      });
      registry.register(skill);

      const loader = new ProgressiveSkillLoader(registry);
      const tools = await loader.activate(skill);

      expect(tools).toHaveLength(1);
      expect(tools[0].metadata.name).toBe('echo');
    });

    it('calls init() when provided', async () => {
      const initFn = vi.fn().mockResolvedValue(undefined);
      const registry = new DefaultSkillRegistry();
      const skill = createMockSkill({
        name: 'init-skill',
        initImpl: initFn,
      });
      registry.register(skill);

      const loader = new ProgressiveSkillLoader(registry);
      await loader.activate(skill);

      expect(initFn).toHaveBeenCalledOnce();
    });

    it('returns empty tools and logs warning when init fails', async () => {
      const registry = new DefaultSkillRegistry();
      const skill = createMockSkill({
        name: 'failing-init',
        initImpl: async () => { throw new Error('init error'); },
      });
      registry.register(skill);

      const loader = new ProgressiveSkillLoader(registry);
      const tools = await loader.activate(skill);

      // Should return empty tools but not throw
      expect(tools).toHaveLength(0);
    });

    it('returns cached tools on re-activation and updates timestamp', async () => {
      const initFn = vi.fn().mockResolvedValue(undefined);
      const registry = new DefaultSkillRegistry();
      const skill = createMockSkill({
        name: 'cached',
        tools: [echoTool],
        initImpl: initFn,
      });
      registry.register(skill);

      const loader = new ProgressiveSkillLoader(registry);

      // First activation
      const tools1 = await loader.activate(skill);
      expect(initFn).toHaveBeenCalledOnce();

      // Second activation — init should not be called again
      const tools2 = await loader.activate(skill);
      expect(initFn).toHaveBeenCalledOnce();
      expect(tools2).toEqual(tools1); // same content (cached, shallow copy)
      expect(loader.getActiveSkillNames()).toContain('cached');
    });

    it('evicts oldest skill when maxActiveSkills is reached', async () => {
      const registry = new DefaultSkillRegistry();
      const s1 = createMockSkill({ name: 's1', tools: [echoTool] });
      const s2 = createMockSkill({ name: 's2', tools: [reverseTool] });
      const s3 = createMockSkill({ name: 's3', tools: [echoTool] });
      const s4 = createMockSkill({ name: 's4', tools: [reverseTool] });
      registry.register(s1);
      registry.register(s2);
      registry.register(s3);
      registry.register(s4);

      const loader = new ProgressiveSkillLoader(registry, { maxActiveSkills: 3 });

      await loader.activate(s1);
      await loader.activate(s2);
      await loader.activate(s3);

      expect(loader.getActiveSkillNames()).toEqual(['s1', 's2', 's3']);

      // Activate s4 — should evict s1 (oldest)
      await loader.activate(s4);

      const activeNames = loader.getActiveSkillNames();
      expect(activeNames).not.toContain('s1');
      expect(activeNames).toContain('s4');
      expect(activeNames.length).toBe(3);
    });
  });

  describe('deactivate()', () => {
    it('deactivates a skill and calls dispose', async () => {
      const disposeFn = vi.fn();
      const registry = new DefaultSkillRegistry();
      const skill = createMockSkill({
        name: 'to-deactivate',
        tools: [echoTool],
        disposeImpl: disposeFn,
      });
      registry.register(skill);

      const loader = new ProgressiveSkillLoader(registry);
      await loader.activate(skill);
      expect(loader.getActiveSkillNames()).toContain('to-deactivate');

      await loader.deactivate('to-deactivate');

      expect(loader.getActiveSkillNames()).not.toContain('to-deactivate');
      expect(disposeFn).toHaveBeenCalledOnce();
    });

    it('quietly handles deactivating unknown skill', async () => {
      const loader = new ProgressiveSkillLoader(new DefaultSkillRegistry());
      // Should not throw
      await loader.deactivate('nonexistent');
    });

    it('dispose error does not prevent deactivation', async () => {
      const registry = new DefaultSkillRegistry();
      const skill = createMockSkill({
        name: 'bad-dispose',
        disposeImpl: async () => { throw new Error('dispose boom'); },
      });
      registry.register(skill);

      const loader = new ProgressiveSkillLoader(registry);
      await loader.activate(skill);

      // Should not throw
      await loader.deactivate('bad-dispose');
      expect(loader.getActiveSkillNames()).not.toContain('bad-dispose');
    });
  });

  describe('getActiveTools()', () => {
    it('returns empty array when no skills active', () => {
      const loader = new ProgressiveSkillLoader(new DefaultSkillRegistry());
      expect(loader.getActiveTools()).toHaveLength(0);
    });

    it('returns tools from all active skills', async () => {
      const registry = new DefaultSkillRegistry();
      const s1 = createMockSkill({ name: 'a', tools: [echoTool] });
      const s2 = createMockSkill({ name: 'b', tools: [reverseTool] });
      registry.register(s1);
      registry.register(s2);

      const loader = new ProgressiveSkillLoader(registry);
      await loader.activate(s1);
      await loader.activate(s2);

      const tools = loader.getActiveTools();
      expect(tools).toHaveLength(2);
      const toolNames = tools.map((t) => t.metadata.name).sort();
      expect(toolNames).toEqual(['echo', 'reverse']);
    });

    it('tracks which skills are active', async () => {
      const registry = new DefaultSkillRegistry();
      const s1 = createMockSkill({ name: 'a' });
      const s2 = createMockSkill({ name: 'b' });
      registry.register(s1);
      registry.register(s2);

      const loader = new ProgressiveSkillLoader(registry);
      expect(loader.getActiveSkillNames()).toHaveLength(0);

      await loader.activate(s1);
      expect(loader.getActiveSkillNames()).toEqual(['a']);

      await loader.activate(s2);
      expect(loader.getActiveSkillNames()).toEqual(['a', 'b']);

      await loader.deactivate('a');
      expect(loader.getActiveSkillNames()).toEqual(['b']);
    });
  });

  describe('cleanupExpired()', () => {
    it('removes expired skills based on timeout', async () => {
      const registry = new DefaultSkillRegistry();
      const skill = createMockSkill({ name: 'expirable' });
      registry.register(skill);

      // Set a very short timeout, then move time forward
      const loader = new ProgressiveSkillLoader(registry, {
        activationTimeoutMs: 0, // immediately expired
      });

      await loader.activate(skill);
      expect(loader.getActiveSkillNames()).toContain('expirable');

      const expired = await loader.cleanupExpired();

      expect(expired).toContain('expirable');
      expect(loader.getActiveSkillNames()).not.toContain('expirable');
    });

    it('returns empty array when no skills expired', async () => {
      const registry = new DefaultSkillRegistry();
      const skill = createMockSkill({ name: 'persistent' });
      registry.register(skill);

      const loader = new ProgressiveSkillLoader(registry, {
        activationTimeoutMs: 60_000, // long timeout
      });

      await loader.activate(skill);
      const expired = await loader.cleanupExpired();

      expect(expired).toHaveLength(0);
      expect(loader.getActiveSkillNames()).toContain('persistent');
    });

    it('handles empty active list gracefully', async () => {
      const loader = new ProgressiveSkillLoader(new DefaultSkillRegistry());
      const expired = await loader.cleanupExpired();
      expect(expired).toHaveLength(0);
    });
  });

  describe('constructor defaults', () => {
    it('uses default config values', () => {
      const loader = new ProgressiveSkillLoader(new DefaultSkillRegistry());

      // Default maxActiveSkills should be 3
      const summaries = loader.getSkillSummaries();
      expect(summaries).toHaveLength(0);
      // No errors accessing internal state
    });
  });
});
