/**
 * 记忆模块 — 导出
 *
 * 后续系列可以通过替换 InMemoryLongTermMemory 为 Redis/MySQL 实现来切换后端。
 */

export { InMemoryLongTermMemory } from './long-term-memory.js';
export type { ConsolidationConfig, ExtractedFact, ConsolidationStrategy } from './consolidation.js';
export { SimpleConsolidationStrategy } from './consolidation.js';
export type { DefaultMemorySystemConfig } from './default-memory-system.js';
export { DefaultMemorySystem } from './default-memory-system.js';
