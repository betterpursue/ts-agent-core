/**
 * Skill 模块 — 核心接口定义
 *
 * 设计原则：
 * - Skill 是 Tool 的上层封装，一个 Skill 包含一个或多个 Tool
 * - 支持 Progressive Disclosure：先暴露 Skill 摘要，按需加载详细工具
 * - Skill 可独立初始化（init）和清理（dispose），支持懒加载和卸载
 * - 向下兼容：现有 Tool 系统完全保留，Skill 是可选的增强层
 */

import type { Tool, ToolMetadata } from '../core/tool.js';

// ─── Skill 元数据 ────────────────────────────────────────────────

/**
 * Skill 元数据 —— 用于 LLM 的 skill 发现和选择
 *
 * 类比：ToolMetadata 是给 LLM 看"这个工具能做什么"，
 * SkillMetadata 是给 LLM 看"这个技能组能解决哪类问题"。
 */
export interface SkillMetadata {
  /** Skill 唯一标识 */
  name: string;
  /** 一句话描述这个 Skill 的用途 */
  description: string;
  /** 版本号 */
  version: string;
  /** 标签 —— 用于匹配和分类 */
  tags: string[];
  /** 这个 Skill 包含的工具名称列表 */
  toolNames: string[];
}

// ─── Skill 接口 ──────────────────────────────────────────────────

/**
 * Skill —— 一组相关工具的容器
 *
 * 核心职责：
 * 1. 声明自己的元数据（名称、描述、标签）
 * 2. 提供包含的 Tool 列表
 * 3. 可选的生命周期管理（init/dispose）
 *
 * 使用方式：
 * ```typescript
 * const skill: Skill = {
 *   metadata: {
 *     name: 'web-search',
 *     description: '搜索网页获取实时信息',
 *     version: '1.0.0',
 *     tags: ['search', 'web', 'research'],
 *     toolNames: ['search', 'fetch'],
 *   },
 *   getTools: () => [SearchTool, WebFetchTool],
 * };
 * ```
 */
export interface Skill {
  readonly metadata: SkillMetadata;

  /**
   * 获取这个 Skill 包含的所有 Tool
   *
   * 注意：返回的是 Tool 实例，不是 ToolMetadata。
   * Tool 实例包含 execute 方法，可以在需要时直接调用。
   */
  getTools(): Tool[];

  /**
   * 初始化 Skill（可选）
   *
   * 在 Skill 首次激活时调用。用于：
   * - 建立连接（数据库、API）
   * - 预加载资源
   * - 验证配置
   *
   * 如果不依赖外部资源，可以不实现此方法。
   */
  init?(): Promise<void>;

  /**
   * 清理 Skill（可选）
   *
   * 在 Skill 被卸载时调用。用于：
   * - 关闭连接
   * - 释放缓存
   * - 清理临时文件
   */
  dispose?(): Promise<void>;
}

// ─── Skill 注册表 ────────────────────────────────────────────────

/**
 * Skill 注册表 —— 管理所有可用 Skill
 *
 * 类比 ToolRegistry：
 * - ToolRegistry 管理单个工具
 * - SkillRegistry 管理工具组（Skill）
 *
 * Agent 通过 SkillRegistry 发现和加载 Skill，
 * 而不是一次性把所有工具的 schema 塞进 context。
 */
export interface SkillRegistry {
  /** 注册一个 Skill */
  register(skill: Skill): void;

  /** 根据名称获取 Skill */
  get(name: string): Skill | undefined;

  /** 列出所有已注册 Skill 的元数据（用于构建 skill 摘要） */
  list(): SkillMetadata[];

  /** 移除 Skill（会 await dispose） */
  unregister(name: string): Promise<boolean>;
}

/**
 * 默认 Skill 注册表实现
 *
 * 内部用 Map 存储，保证 O(1) 查找。
 * 注册时检查重复，避免同名 Skill 覆盖。
 */
export class DefaultSkillRegistry implements SkillRegistry {
  private skills = new Map<string, Skill>();

  register(skill: Skill): void {
    const name = skill.metadata.name;
    if (this.skills.has(name)) {
      throw new Error(
        `Skill '${name}' is already registered. ` +
        `Unregister the existing one first, or use a different name.`
      );
    }
    this.skills.set(name, skill);
  }

  get(name: string): Skill | undefined {
    return this.skills.get(name);
  }

  list(): SkillMetadata[] {
    return Array.from(this.skills.values()).map((s) => s.metadata);
  }

  async unregister(name: string): Promise<boolean> {
    const skill = this.skills.get(name);
    if (skill) {
      if (skill.dispose) {
        try {
          await skill.dispose();
        } catch (err) {
          console.warn(
            `[DefaultSkillRegistry] Error disposing skill '${name}':`,
            err instanceof Error ? err.message : String(err)
          );
        }
      }
    }
    return this.skills.delete(name);
  }

  /** 获取注册表中的 Skill 数量 */
  get size(): number {
    return this.skills.size;
  }
}

// ─── Progressive Loader ──────────────────────────────────────────

/**
 * Skill 摘要 —— 用于 LLM 的 skill 选择
 *
 * 比完整 SkillMetadata 更精简，只包含 LLM 决策需要的信息。
 */
export interface SkillSummary {
  name: string;
  description: string;
  tags: string[];
}

/**
 * ProgressiveSkillLoader 配置
 */
export interface ProgressiveLoaderConfig {
  /** 最多同时激活的 Skill 数量（防止 context 膨胀） */
  maxActiveSkills?: number;
  /** Skill 激活超时（ms），超时后自动卸载 */
  activationTimeoutMs?: number;
  /** 选择策略 */
  selectionStrategy?: 'relevance' | 'manual';
}

/**
 * 已激活的 Skill 记录
 */
interface ActiveSkillRecord {
  skill: Skill;
  activatedAt: number;
  tools: Tool[];
}

/**
 * 渐进式 Skill 加载器
 *
 * 核心职责：
 * 1. 根据当前对话内容，从注册表中选择最相关的 Skill
 * 2. 按需激活 Skill（调用 init、缓存 Tools）
 * 3. 当激活数量超限时，淘汰最久未使用的 Skill
 * 4. 提供当前激活 Skill 的工具列表，供 Agent 注入到 LLM context
 *
 * Progressive Disclosure 的工程化实现：
 * - 第一层：LLM 只看到 skill 摘要列表（几十个 token）
 * - 第二层：选中 skill 后，才加载详细工具 schema（几百个 token）
 * - 避免一次性把所有工具的 schema 全塞进 context
 */
export class ProgressiveSkillLoader {
  private registry: SkillRegistry;
  private config: Required<ProgressiveLoaderConfig>;
  private activeSkills = new Map<string, ActiveSkillRecord>();

  constructor(registry: SkillRegistry, config?: ProgressiveLoaderConfig) {
    this.registry = registry;
    this.config = {
      maxActiveSkills: config?.maxActiveSkills ?? 3,
      activationTimeoutMs: config?.activationTimeoutMs ?? 30 * 60 * 1000, // 30 分钟
      selectionStrategy: config?.selectionStrategy ?? 'relevance',
    };
  }

  /**
   * 获取所有 Skill 的摘要（用于构建 LLM 可读的 skill 列表）
   */
  getSkillSummaries(): SkillSummary[] {
    return this.registry.list().map((meta) => ({
      name: meta.name,
      description: meta.description,
      tags: meta.tags,
    }));
  }

  /**
   * 根据查询文本选择最相关的 Skill
   *
   * 选择策略：
   * - 'relevance'：基于查询文本与 Skill 名称/标签/描述的关键词匹配度排序
   * - 'manual'：返回所有已注册 Skill，由外部手动指定激活
   *
   * 匹配维度：
   * 1. Skill 名称包含查询词（权重 2）
   * 2. 标签匹配查询词（权重 1）
   * 3. 描述中的关键词匹配（权重 0.5）
   */
  selectSkills(query: string): Skill[] {
    const allSkills = this.registry.list();
    const scored = allSkills.map((meta) => {
      const skill = this.registry.get(meta.name)!;
      const score = this.computeRelevance(meta, query);
      return { skill, score };
    });

    scored.sort((a, b) => b.score - a.score);

    return scored
      .filter((s) => s.score > 0)
      .slice(0, this.config.maxActiveSkills)
      .map((s) => s.skill);
  }

  /**
   * 激活一个 Skill
   *
   * 激活流程：
   * 1. 检查是否已经激活，如果是直接返回缓存的工具列表
   * 2. 如果超出上限，淘汰最久未使用的 Skill
   * 3. 调用 skill.init()（如果存在）
   * 4. 调用 skill.getTools() 缓存工具列表
   *
   * @returns 激活 Skill 包含的 Tool 列表
   */
  async activate(skill: Skill): Promise<Tool[]> {
    // 已经激活，直接返回缓存（浅拷贝防止外部修改缓存）
    const existing = this.activeSkills.get(skill.metadata.name);
    if (existing) {
      existing.activatedAt = Date.now();
      return [...existing.tools];
    }

    // 超出上限，淘汰最老的
    if (this.activeSkills.size >= this.config.maxActiveSkills) {
      this.evictOldest();
    }

    // 初始化（如果 Skill 需要）
    if (skill.init) {
      try {
        await skill.init();
      } catch (err) {
        console.warn(
          `[ProgressiveSkillLoader] Failed to init skill '${skill.metadata.name}':`,
          err instanceof Error ? err.message : String(err)
        );
        // init 失败不影响注册，但返回空工具列表
        return [];
      }
    }

    const tools = skill.getTools();
    this.activeSkills.set(skill.metadata.name, {
      skill,
      activatedAt: Date.now(),
      tools,
    });

    // 浅拷贝返回，防止外部篡改缓存
    return [...tools];
  }

  /**
   * 停用一个 Skill
   *
   * 调用 skill.dispose() 释放资源，从激活列表移除。
   * 如果 Skill 不在激活列表，静默跳过。
   */
  async deactivate(name: string): Promise<void> {
    const record = this.activeSkills.get(name);
    if (record) {
      if (record.skill.dispose) {
        try {
          await record.skill.dispose();
        } catch (err) {
          console.warn(
            `[ProgressiveSkillLoader] Error disposing skill '${name}':`,
            err instanceof Error ? err.message : String(err)
          );
        }
      }
      this.activeSkills.delete(name);
    }
  }

  /**
   * 获取当前所有激活 Skill 的工具列表
   *
   * Agent 可以在每次 LLM 调用前，把这个列表和 ToolRegistry 中的工具合并，
   * 实现"当前激活 Skill 的工具优先展示"的效果。
   */
  getActiveTools(): Tool[] {
    const all: Tool[] = [];
    for (const record of this.activeSkills.values()) {
      all.push(...record.tools);
    }
    return all;
  }

  /**
   * 获取当前激活的 Skill 名称列表
   */
  getActiveSkillNames(): string[] {
    return Array.from(this.activeSkills.keys());
  }

  /**
   * 清理超时的 Skill
   *
   * 定期调用此方法，清理长时间未使用的 Skill，释放资源。
   */
  async cleanupExpired(): Promise<string[]> {
    const now = Date.now();
    const expired: string[] = [];

    for (const [name, record] of this.activeSkills.entries()) {
      if (now - record.activatedAt >= this.config.activationTimeoutMs) {
        expired.push(name);
      }
    }

    for (const name of expired) {
      await this.deactivate(name);
    }

    return expired;
  }

  /**
   * 根据查询文本自动选择并激活相关 Skill
   *
   * 这是给 Agent 的便捷入口：一句调用完成「选择 + 激活」两步。
   * 激活后通过 getActiveTools() 获取工具列表。
   * 同时自动停用当前激活但本次未被选中的 Skill。
   */
  async selectAndActivate(query: string): Promise<void> {
    const selected = this.selectSkills(query);

    // 去重激活：已经激活的不重复 init，只刷新时间戳
    for (const skill of selected) {
      if (this.activeSkills.has(skill.metadata.name)) {
        this.activeSkills.get(skill.metadata.name)!.activatedAt = Date.now();
      } else {
        await this.activate(skill);
      }
    }

    // 停用不相关的 Skill：当前激活但未被本次查询选中的
    const selectedNames = new Set(selected.map((s) => s.metadata.name));
    for (const name of this.getActiveSkillNames()) {
      if (!selectedNames.has(name)) {
        await this.deactivate(name);
      }
    }
  }

  /**
   * 获取当前已激活工具的元数据（用于构建 LLM 的 function calling schema）
   */
  getActiveToolMetadatas(): import('../core/tool.js').ToolMetadata[] {
    const tools = this.getActiveTools();
    return tools.map((t) => t.metadata);
  }

  // ─── 私有方法 ──────────────────────────────────────────────────

  /**
   * 计算 Skill 与查询文本的相关度分数
   */
  private computeRelevance(skill: SkillMetadata, query: string): number {
    const q = query.toLowerCase();
    const qWords = q.split(/\s+/).filter((w) => w.length > 1);
    if (qWords.length === 0) return 0;

    let score = 0;

    // 1. 名称匹配（权重最高）
    if (skill.name.toLowerCase().includes(q)) {
      score += 2;
    }

    // 2. 标签匹配
    for (const tag of skill.tags) {
      const tagLower = tag.toLowerCase();
      if (q.includes(tagLower)) {
        score += 1;
      }
      // 标签包含查询词
      for (const qw of qWords) {
        if (tagLower.includes(qw)) {
          score += 0.5;
        }
      }
    }

    // 3. 描述关键词匹配
    const descLower = skill.description.toLowerCase();
    for (const qw of qWords) {
      if (descLower.includes(qw)) {
        score += 0.5;
      }
    }

    return score;
  }

  /**
   * 淘汰最久未使用的 Skill
   */
  private async evictOldest(): Promise<void> {
    let oldestName: string | null = null;
    let oldestTime = Infinity;

    for (const [name, record] of this.activeSkills.entries()) {
      if (record.activatedAt < oldestTime) {
        oldestTime = record.activatedAt;
        oldestName = name;
      }
    }

    if (oldestName) {
      console.log(
        `[ProgressiveSkillLoader] Evicting skill '${oldestName}' (oldest active)`
      );
      await this.deactivate(oldestName);
    }
  }
}
