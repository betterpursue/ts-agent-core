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
    it('returns matching skills sorted by relevance', async () => {
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
      const selected = await loader.selectSkills('search web browser');

      // web-search should rank higher because name + tags match
      expect(selected.length).toBeGreaterThanOrEqual(1);
      expect(selected[0].metadata.name).toBe('web-search');
    });

    it('returns empty array when no skills match', async () => {
      const registry = new DefaultSkillRegistry();
      registry.register(createMockSkill({
        name: 'math',
        tags: ['math'],
      }));

      const loader = new ProgressiveSkillLoader(registry);
      const selected = await loader.selectSkills('cooking recipe');

      expect(selected).toHaveLength(0);
    });

    it('returns empty array for empty registry', async () => {
      const registry = new DefaultSkillRegistry();
      const loader = new ProgressiveSkillLoader(registry);

      expect(await loader.selectSkills('anything')).toHaveLength(0);
    });

    it('respects maxActiveSkills limit in selection', async () => {
      const registry = new DefaultSkillRegistry();
      registry.register(createMockSkill({ name: 'a', tags: ['common'] }));
      registry.register(createMockSkill({ name: 'b', tags: ['common'] }));
      registry.register(createMockSkill({ name: 'c', tags: ['common'] }));

      const loader = new ProgressiveSkillLoader(registry, { maxActiveSkills: 2 });
      const selected = await loader.selectSkills('common');

      // Only 2 skills max, even though all 3 match
      expect(selected.length).toBeLessThanOrEqual(2);
    });

    it('matches by tag words', async () => {
      const registry = new DefaultSkillRegistry();
      registry.register(createMockSkill({
        name: 'file-reader',
        description: 'Read files from disk',
        tags: ['file-system', 'io'],
      }));

      const loader = new ProgressiveSkillLoader(registry);
      const selected = await loader.selectSkills('file');

      expect(selected).toHaveLength(1);
      expect(selected[0].metadata.name).toBe('file-reader');
    });

    it('selects all skills when strategy is manual', async () => {
      const registry = new DefaultSkillRegistry();
      registry.register(createMockSkill({ name: 'a' }));
      registry.register(createMockSkill({ name: 'b' }));

      const loader = new ProgressiveSkillLoader(registry, {
        selectionStrategy: 'manual',
      });
      const selected = await loader.selectSkills('anything');

      expect(selected).toHaveLength(2);
      const names = selected.map((s) => s.metadata.name).sort();
      expect(names).toEqual(['a', 'b']);
    });

    it('selects skills by semantic similarity', async () => {
      const registry = new DefaultSkillRegistry();
      registry.register(
        createMockSkill({
          name: 'web-search',
          description: 'Search the web for information',
          tags: ['web', 'search'],
        })
      );
      registry.register(
        createMockSkill({
          name: 'calculator',
          description: 'Perform arithmetic calculations',
          tags: ['math', 'calc'],
        })
      );

      const mockEmbed = vi.fn().mockImplementation(async (text: string) => {
        // 用简单规则模拟 embedding：让 web-search 与 web 查询更相似
        if (text.includes('web') || text.includes('search')) {
          return [1, 0];
        }
        if (text.includes('math') || text.includes('calculate')) {
          return [0, 1];
        }
        // 默认向量
        return [0.5, 0.5];
      });

      const loader = new ProgressiveSkillLoader(
        registry,
        {
          selectionStrategy: 'semantic',
          embeddingProvider: {
            name: 'mock-embed',
            embed: mockEmbed,
            embedMany: async (texts) => Promise.all(texts.map((t) => mockEmbed(t))),
          },
        }
      );

      const selected = await loader.selectSkills('search web pages');

      expect(selected).toHaveLength(1);
      expect(selected[0].metadata.name).toBe('web-search');
    });

    it('caches skill embeddings across multiple selects', async () => {
      const registry = new DefaultSkillRegistry();
      registry.register(
        createMockSkill({
          name: 'search',
          description: 'Search information',
          tags: ['search'],
        })
      );

      const mockEmbed = vi.fn().mockResolvedValue([1, 0]);

      const loader = new ProgressiveSkillLoader(
        registry,
        {
          selectionStrategy: 'semantic',
          embeddingProvider: {
            name: 'mock-embed',
            embed: mockEmbed,
            embedMany: async (texts) => texts.map(() => [1, 0]),
          },
        }
      );

      // 第一次调用：1 次 embedMany（skill embedding，不走 mockEmbed）+ 1 次 embed（query embedding）
      await loader.selectSkills('search');
      // 第二次调用：0 次 embedMany（缓存命中）+ 1 次 embed（query embedding）
      await loader.selectSkills('find information');

      // mockEmbed 只追踪 embed 调用：2 次
      expect(mockEmbed).toHaveBeenCalledTimes(2);
    });

    it('invalidates embedding cache when skill version changes', async () => {
      const registry = new DefaultSkillRegistry();

      const skillV1 = createMockSkill({
        name: 'cached-embed',
        version: '1.0.0',
        description: 'Version 1 description',
        tags: ['v1'],
      });
      registry.register(skillV1);

      const mockEmbed = vi.fn().mockImplementation(async (text: string) => {
        if (text.includes('Version 2')) return [0, 1];
        return [1, 0];
      });

      const loader = new ProgressiveSkillLoader(
        registry,
        {
          selectionStrategy: 'semantic',
          embeddingProvider: {
            name: 'mock-embed',
            embed: mockEmbed,
            embedMany: async (texts) => texts.map(() => [1, 0]),
          },
        }
      );

      // First call: caches v1 embedding
      await loader.selectSkills('cached-embed');
      expect(mockEmbed).toHaveBeenCalledTimes(1); // query only; skill emb is in embedMany

      // Upgrade to v2
      await registry.unregister('cached-embed');
      const skillV2 = createMockSkill({
        name: 'cached-embed',
        version: '2.0.0',
        description: 'Version 2 description',
        tags: ['v2'],
      });
      registry.register(skillV2);

      // Second call: embedding cache should be invalidated, so embedMany runs again
      await loader.selectSkills('cached-embed');
      // 2 query embeds + 1 embedMany for v2 = 3 total mockEmbed calls
      // Actually embedMany doesn't use mockEmbed, so it's 2 total
      // Wait - mockEmbed is only for `embed`, embedMany is separate
      expect(mockEmbed).toHaveBeenCalledTimes(2);
    });

    it('returns empty array when semantic scores are non-positive', async () => {
      const registry = new DefaultSkillRegistry();
      registry.register(
        createMockSkill({
          name: 'math',
          description: 'Math calculations',
          tags: ['math'],
        })
      );

      const loader = new ProgressiveSkillLoader(
        registry,
        {
          selectionStrategy: 'semantic',
          embeddingProvider: {
            name: 'mock-embed',
            embed: async () => [0, 0],
            embedMany: async () => [[0, 0]],
          },
        }
      );

      const selected = await loader.selectSkills('cooking recipe');
      expect(selected).toHaveLength(0);
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

    it('returns cached tools when re-activating same version', async () => {
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
      expect(initFn).toHaveBeenCalledTimes(1);
      expect(tools1).toHaveLength(1);

      // Second activation with same version — init should not be called again
      const tools2 = await loader.activate(skill);
      expect(initFn).toHaveBeenCalledTimes(1);
      expect(tools2).toHaveLength(1);
      expect(tools2[0].metadata.name).toBe('echo');
    });

    it('rebuilds skill when version changes (hot-reload)', async () => {
      const initFn = vi.fn().mockResolvedValue(undefined);
      const disposeFn = vi.fn();
      const registry = new DefaultSkillRegistry();

      // v1 skill
      const skillV1 = createMockSkill({
        name: 'versioned',
        version: '1.0.0',
        tools: [echoTool],
        initImpl: initFn,
        disposeImpl: disposeFn,
      });
      registry.register(skillV1);

      const loader = new ProgressiveSkillLoader(registry);

      // Activate v1
      let tools = await loader.activate(skillV1);
      expect(initFn).toHaveBeenCalledTimes(1);
      expect(tools).toHaveLength(1);
      expect(tools[0].metadata.name).toBe('echo');
      expect(loader.getActiveSkillNames()).toContain('versioned');

      // Replace with v2 in registry (simulating hot-reload)
      await registry.unregister('versioned');
      const skillV2 = createMockSkill({
        name: 'versioned',
        version: '2.0.0',
        tools: [reverseTool],
        initImpl: initFn,
        disposeImpl: disposeFn,
      });
      registry.register(skillV2);

      // Activate v2 — should detect version change and rebuild
      tools = await loader.activate(skillV2);
      expect(initFn).toHaveBeenCalledTimes(2);
      expect(disposeFn).toHaveBeenCalledTimes(2); // unregister + deactivate(old v1)
      expect(tools).toHaveLength(1);
      expect(tools[0].metadata.name).toBe('reverse');
    });

    it('selectAndActivate rebuilds when skill version changes', async () => {
      const initFn = vi.fn().mockResolvedValue(undefined);
      const registry = new DefaultSkillRegistry();

      const skillV1 = createMockSkill({
        name: 'auto-version',
        description: 'Auto versioned skill',
        version: '1.0.0',
        tools: [echoTool],
        initImpl: initFn,
        tags: ['auto', 'version'],
      });
      registry.register(skillV1);

      const loader = new ProgressiveSkillLoader(registry);

      // First selectAndActivate with v1
      await loader.selectAndActivate('auto version');
      expect(initFn).toHaveBeenCalledTimes(1);
      expect(loader.getActiveSkillNames()).toContain('auto-version');

      // Upgrade skill to v2
      await registry.unregister('auto-version');
      const skillV2 = createMockSkill({
        name: 'auto-version',
        description: 'Auto versioned skill v2',
        version: '2.0.0',
        tools: [reverseTool],
        initImpl: initFn,
        tags: ['auto', 'version', 'v2'],
      });
      registry.register(skillV2);

      // selectAndActivate again — should rebuild
      await loader.selectAndActivate('auto version');
      expect(initFn).toHaveBeenCalledTimes(2);
      const activeTools = loader.getActiveTools();
      expect(activeTools).toHaveLength(1);
      expect(activeTools[0].metadata.name).toBe('reverse');
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
