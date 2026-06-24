/**
 * Skill 模块导出入口
 *
 * 所有 Skill 类型和实现集中导出。
 * import { Skill, SkillRegistry, DefaultSkillRegistry } from 'ts-agent-core/skill'
 */
export type {
  SkillMetadata,
  Skill,
  SkillRegistry,
  SkillSummary,
  ProgressiveLoaderConfig,
} from './skill.js';

export {
  DefaultSkillRegistry,
  ProgressiveSkillLoader,
} from './skill.js';

export type {
  ToSkillOptions,
} from './tool-adapter.js';

export {
  toSkill,
  groupToolsByPrefix,
  ToolRegistryAdapter,
} from './tool-adapter.js';
