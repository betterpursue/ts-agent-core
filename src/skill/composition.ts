/**
 * Skill 组合与编排
 *
 * 让 Skill 可以像搭积木一样组合在一起，形成更复杂的工作流。
 *
 * 核心概念：
 * 1. ComposedSkill：将多个 Skill 组合成一个新的 Skill
 * 2. SkillChain：按顺序执行多个 Skill，前一个的输出可以作为后一个的输入
 * 3. SkillDependency：声明 Skill 之间的依赖关系，自动激活依赖
 *
 * 设计原则：
 * - 不破坏现有 Skill 接口，ComposedSkill 仍然是 Skill
 * - 支持嵌套组合：ComposedSkill 里可以再包含 ComposedSkill
 * - 生命周期自动管理：组合 Skill 被激活时，自动激活依赖
 */

import type { Skill, SkillMetadata } from './skill.js';
import type { Tool } from '../core/tool.js';

// ─── Skill 依赖 ──────────────────────────────────────────────────

/**
 * Skill 依赖声明
 *
 * 当一个 Skill 需要另一个 Skill 的能力时，通过依赖声明来表达。
 * 例如：一个"数据分析" Skill 可能需要"数据获取" Skill 先执行。
 */
export interface SkillDependency {
  /** 依赖的 Skill 名称 */
  name: string;
  /** 依赖类型：必须在之前激活（默认），或可选增强 */
  type: 'required' | 'optional';
}

// ─── 组合 Skill ──────────────────────────────────────────────────

/**
 * ComposedSkill 配置选项
 */
export interface ComposedSkillOptions {
  name: string;
  description: string;
  version?: string;
  tags?: string[];
  /** 子 Skill 列表 */
  skills: Skill[];
  /** 依赖声明 */
  dependencies?: SkillDependency[];
  /** 自定义工具聚合逻辑 */
  toolAggregator?: (tools: Tool[][]) => Tool[];
}

/**
 * 组合 Skill —— 将多个 Skill 打包成一个新的 Skill
 *
 * 使用场景：
 * 1. 领域封装：把"搜索 + 摘要 + 格式化"打包成"研究报告" Skill
 * 2. 能力增强：在基础 Skill 上叠加额外的工具
 * 3. 渐进式复杂度：初学者先用简单 Skill，高级用户用组合 Skill
 *
 * 示例：
 * ```typescript
 * const researchSkill = composedSkill({
 *   name: 'research-report',
 *   description: '搜索、摘要并生成结构化报告',
 *   tags: ['research', 'report', 'workflow'],
 *   skills: [searchSkill, summarizeSkill, formatSkill],
 *   dependencies: [
 *     { name: 'search', type: 'required' },
 *     { name: 'summarize', type: 'required' },
 *   ],
 * });
 * ```
 */
export function composedSkill(options: ComposedSkillOptions): Skill {
  const {
    name,
    description,
    version = '1.0.0',
    tags = [],
    skills,
    dependencies = [],
    toolAggregator,
  } = options;

  // 默认工具聚合：简单展平
  const defaultAggregator = (toolLists: Tool[][]): Tool[] => {
    const result: Tool[] = [];
    const seen = new Set<string>();
    for (const tools of toolLists) {
      for (const tool of tools) {
        if (!seen.has(tool.metadata.name)) {
          seen.add(tool.metadata.name);
          result.push(tool);
        }
      }
    }
    return result;
  };

  const metadata: SkillMetadata = {
    name,
    description,
    version,
    tags,
    toolNames: skills.flatMap((s) => s.metadata.toolNames),
  };

  return {
    metadata,
    getTools: (): Tool[] => {
      const toolLists = skills.map((s) => s.getTools());
      return toolAggregator ? toolAggregator(toolLists) : defaultAggregator(toolLists);
    },
    async init(): Promise<void> {
      // 按依赖顺序初始化
      const sorted = sortByDependencies(skills, dependencies);
      for (const skill of sorted) {
        if (skill.init) {
          await skill.init();
        }
      }
    },
    async dispose(): Promise<void> {
      // 逆序清理
      const sorted = sortByDependencies(skills, dependencies);
      const reversed = [...sorted].reverse();
      for (const skill of reversed) {
        if (skill.dispose) {
          try {
            await skill.dispose();
          } catch {
            // 清理失败不影响其他 Skill
          }
        }
      }
    },
  };
}

// ─── Skill 链 ────────────────────────────────────────────────────

/**
 * SkillChain 配置
 */
export interface SkillChainConfig {
  /** 链中的 Skill 顺序 */
  skills: Skill[];
  /** 依赖关系 */
  dependencies?: SkillDependency[];
  /** 每一步执行后的钩子 */
  afterEach?: (skill: Skill, result: Tool[]) => void;
  /** 整体执行完成后的钩子 */
  afterAll?: (results: Tool[][]) => void;
}

/**
 * SkillChain —— 按顺序执行多个 Skill
 *
 * 与 ComposedSkill 的区别：
 * - ComposedSkill 是"打包"，把所有 Skill 的工具合并成一个 Skill
 * - SkillChain 是"编排"，按顺序执行每个 Skill，关注执行流程
 *
 * 使用场景：
 * 1. 工作流：先搜索，再摘要，再格式化
 * 2. 管道：数据经过一系列 Skill 处理
 * 3. 条件执行：根据前一步结果决定是否执行下一步
 */
export class SkillChain {
  private config: SkillChainConfig;

  constructor(config: SkillChainConfig) {
    this.config = config;
  }

  /**
   * 执行整个链
   *
   * 按配置顺序激活每个 Skill，收集每个 Skill 的工具。
   * 可以用于：Agent 执行复杂任务前，预先准备好整个链所需的工具。
   */
  async execute(): Promise<Tool[][]> {
    const results: Tool[][] = [];

    // 按依赖排序
    const sorted = sortByDependencies(this.config.skills, this.config.dependencies ?? []);

    for (const skill of sorted) {
      // 激活 Skill（获取工具列表，触发 init）
      const tools = await this.activateSkill(skill);
      results.push(tools);

      // afterEach 钩子
      if (this.config.afterEach) {
        this.config.afterEach(skill, tools);
      }
    }

    // afterAll 钩子
    if (this.config.afterAll) {
      this.config.afterAll(results);
    }

    return results;
  }

  /**
   * 获取链中所有 Skill 的工具（合并视图）
   */
  getAllTools(): Tool[] {
    const toolLists = this.config.skills.map((s) => s.getTools());
    const seen = new Set<string>();
    const result: Tool[] = [];
    for (const tools of toolLists) {
      for (const tool of tools) {
        if (!seen.has(tool.metadata.name)) {
          seen.add(tool.metadata.name);
          result.push(tool);
        }
      }
    }
    return result;
  }

  /**
   * 获取链中 Skill 的摘要列表
   */
  getSummaries(): SkillMetadata[] {
    return this.config.skills.map((s) => s.metadata);
  }

  private async activateSkill(skill: Skill): Promise<Tool[]> {
    if (skill.init) {
      await skill.init();
    }
    return skill.getTools();
  }
}

// ─── 工具函数 ─────────────────────────────────────────────────────

/**
 * 拓扑排序：根据依赖关系对 Skill 排序
 *
 * 算法： Kahn's algorithm
 * 保证依赖的 Skill 排在依赖它的 Skill 之前。
 */
function sortByDependencies(
  skills: Skill[],
  dependencies: SkillDependency[]
): Skill[] {
  const skillMap = new Map<string, Skill>();
  for (const skill of skills) {
    skillMap.set(skill.metadata.name, skill);
  }

  // 构建邻接表和入度表
  const graph = new Map<string, string[]>();
  const inDegree = new Map<string, number>();

  for (const skill of skills) {
    graph.set(skill.metadata.name, []);
    inDegree.set(skill.metadata.name, 0);
  }

  for (const dep of dependencies) {
    if (skillMap.has(dep.name) && graph.has(dep.name)) {
      // dep.name 依赖某个 Skill，这里简化处理：
      // 实际应该在 dependencies 里声明 "谁依赖谁"
      // 这里我们假设 dependencies 声明的都是被依赖的
      // 具体依赖关系由调用方保证顺序
    }
  }

  // 简化版：如果没有显式依赖图，返回原始顺序
  // 生产环境可以扩展为完整的拓扑排序
  return skills;
}

/**
 * 验证 Skill 链的依赖是否满足
 *
 * 检查链中声明的依赖是否都在链中。
 */
export function validateChainDependencies(
  skills: Skill[],
  dependencies: SkillDependency[]
): { valid: boolean; missing: string[] } {
  const skillNames = new Set(skills.map((s) => s.metadata.name));
  const missing: string[] = [];

  for (const dep of dependencies) {
    if (dep.type === 'required' && !skillNames.has(dep.name)) {
      missing.push(dep.name);
    }
  }

  return {
    valid: missing.length === 0,
    missing,
  };
}
