/**
 * Tool → Skill 适配器
 *
 * 让现有 Tool 能直接包装为 Skill，无需改造成本。
 *
 * 使用场景：
 * 1. 已有工具不想改动代码，直接套个 Skill 壳
 * 2. 按领域对工具进行分组（如 'web-tools'、'math-tools'）
 * 3. 无侵入迁移：保持 ToolRegistry 不变，同时开启 Skill 系统
 */

import type { Skill, SkillMetadata } from './skill.js';
import type { Tool, ToolMetadata } from '../core/tool.js';

/**
 * 创建一个将一组工具聚合为 Skill 的配置选项
 */
export interface ToSkillOptions {
  name: string;
  description: string;
  version?: string;
  tags?: string[];
  tools: Tool[];
  init?: () => Promise<void>;
  dispose?: () => Promise<void>;
}

/**
 * 创建 Skill 配置快捷函数
 *
 * 把一组现有的 Tool 实例包装成一个 Skill。
 * 自动生成 toolNames 等元数据。
 *
 * ```typescript
 * const searchSkill = toSkill({
 *   name: 'web-search',
 *   description: '搜索和获取网页内容',
 *   tags: ['search', 'web', 'research'],
 *   tools: [searchTool, fetchTool],
 * });
 * ```
 */
export function toSkill(options: ToSkillOptions): Skill {
  const metadata: SkillMetadata = {
    name: options.name,
    description: options.description,
    version: options.version ?? '1.0.0',
    tags: options.tags ?? [],
    toolNames: options.tools.map((t) => t.metadata.name),
  };

  return {
    metadata,
    getTools: () => options.tools,
    ...(options.init ? { init: options.init } : {}),
    ...(options.dispose ? { dispose: options.dispose } : {}),
  };
}

/**
 * 批量将多个 Tool 按照命名约定自动分组为 Skill
 *
 * 命名约定：Tool 名称用下划线分段，第一段作为 Skill 分组名。
 * 例如：
 * - web_search, web_fetch → skill 'web' (tools: web_search, web_fetch)
 * - math_calc, math_stats → skill 'math' (tools: math_calc, math_stats)
 * - file_reader → skill 'file' (tools: file_reader)
 *
 * 如果不想用默认描述，可以通过 descriptions 参数自定义。
 *
 * 主要用途：从已存在的 ToolRegistry 快速搭建 Skill 系统，
 * 用于扩展示例或演示场景。
 */
export function groupToolsByPrefix(
  tools: Tool[],
  descriptions?: Record<string, string>
): Skill[] {
  const groups = new Map<string, Tool[]>();

  for (const tool of tools) {
    const prefix = tool.metadata.name.split('_')[0];
    if (!groups.has(prefix)) {
      groups.set(prefix, []);
    }
    groups.get(prefix)!.push(tool);
  }

  const skills: Skill[] = [];
  for (const [prefix, groupTools] of groups.entries()) {
    const desc =
      descriptions?.[prefix] ?? `Provides ${groupTools.length} tools related to ${prefix}`;
    skills.push(
      toSkill({
        name: `${prefix}-tools`,
        description: desc,
        tags: [prefix],
        tools: groupTools,
      })
    );
  }

  return skills;
}

/**
 * 适配器包装 —— 将 Progressive Skill Loader 与 ToolRegistry 桥接
 *
 * 当 Agent 同时配置了 ToolRegistry 和 SkillLoader 时，
 * 需要通过此适配器决定最终暴露给 LLM 的工具列表：
 *
 * 合并策略：
 * 1. Skill 暴露的系统级工具（无标签/通用工具）始终可用
 * 2. Skill 激活时，对应工具加入可用列表
 * 3. 工具如果既有全局注册又在 Skill 中，取 Skill 版本（覆盖）
 */
export class ToolRegistryAdapter {
  private toolMap = new Map<string, Tool>();
  private overridden = new Set<string>();

  /**
   * 注册全局工具（独立于 Skill 的工具）
   */
  registerGlobal(tool: Tool): void {
    this.toolMap.set(tool.metadata.name, tool);
  }

  /**
   * 批量注册全局工具
   */
  registerGlobalMany(tools: Tool[]): void {
    for (const tool of tools) {
      this.registerGlobal(tool);
    }
  }

  /**
   * 标记工具被 Skill 覆盖
   */
  markOverridden(toolName: string): void {
    this.overridden.add(toolName);
  }

  /**
   * 合并全局工具和 Skill 激活工具
   *
   * 返回给 LLM 的元数据列表：
   * 1. 没有对应的 Skill 覆盖的全局工具
   * 2. 所有激活 Skill 暴露的工具
   *
   * 返回完整的 Tool 映射（给 Agent 执行用）：
   * - 优先使用 Skill 层注册的 Tool 实例
   * - 回退到全局注册的 Tool 实例
   */
  merge(
    activeSkillTools: Tool[]
  ): { metadata: ToolMetadata[]; toolMap: Map<string, Tool> } {
    const mergedMap = new Map<string, Tool>();
    const activeNames = new Set<string>();

    // 1. Skill 工具优先
    for (const tool of activeSkillTools) {
      mergedMap.set(tool.metadata.name, tool);
      activeNames.add(tool.metadata.name);
    }

    // 2. 全局工具（未被覆盖的）
    for (const [name, tool] of this.toolMap.entries()) {
      if (!activeNames.has(name)) {
        mergedMap.set(name, tool);
      }
    }

    const metadata = Array.from(mergedMap.values()).map((t) => t.metadata);

    return { metadata, toolMap: mergedMap };
  }

  /**
   * 获取所有已注册的全局工具名称
   */
  getGlobalNames(): string[] {
    return Array.from(this.toolMap.keys());
  }
}
