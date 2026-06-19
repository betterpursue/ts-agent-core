/**
 * 记忆模块 — 导出
 *
 * 后续系列可以通过替换 InMemoryLongTermMemory 为 Redis/MySQL 实现来切换后端。
 * 检索结果重排（时效加权 + MMR）通过 Reranker 接口注入。
 */

export { InMemoryLongTermMemory } from './long-term-memory.js';
export type { ConsolidationConfig, ExtractedFact, ConsolidationStrategy } from './consolidation.js';
export { SimpleConsolidationStrategy } from './consolidation.js';
export type { DefaultMemorySystemConfig } from './default-memory-system.js';
export { DefaultMemorySystem } from './default-memory-system.js';
export type { MemoryInjector } from '../core/memory.js';
export { DefaultMemoryInjector, NoopMemoryInjector } from './memory-injector.js';
export type { Reranker, RerankerConfig } from './reranker.js';
export { DefaultReranker, NoopReranker } from './reranker.js';
