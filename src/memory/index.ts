/**
 * 记忆模块 — 导出
 *
 * 第三篇：基于 Embedding 的语义检索新增：
 * - InMemoryLongTermMemory 支持 SearchMode (keyword / semantic / hybrid)
 * - OpenAIEmbeddingProvider / NoopEmbeddingProvider
 * - cosineSimilarity / l2Normalize 工具函数
 * - VectorIndex 向量索引
 *
 * 第四篇：LLM 驱动的记忆合并新增：
 * - LLMConsolidationStrategy / LLMConsolidationConfig
 * - LLMProvider 接口 / OpenAIProvider / MockLLMProvider
 * - ConsolidationStrategy 接口增加 merge() 方法
 */

export { InMemoryLongTermMemory } from './long-term-memory.js';
export type { SearchMode } from './long-term-memory.js';
export type { ConsolidationConfig, ExtractedFact, ConsolidationStrategy } from './consolidation.js';
export { SimpleConsolidationStrategy } from './consolidation.js';
export { LLMConsolidationStrategy } from './llm-consolidation.js';
export type { LLMConsolidationConfig, ConsolidationLLM } from './llm-consolidation.js';
export { OpenAIConsolidation, MockConsolidationLLM } from './llm-consolidation.js';
export type { DefaultMemorySystemConfig } from './default-memory-system.js';
export { DefaultMemorySystem } from './default-memory-system.js';
export type { MemoryInjector } from '../core/memory.js';
export { DefaultMemoryInjector, NoopMemoryInjector } from './memory-injector.js';
export type { Reranker, RerankerConfig } from './reranker.js';
export { DefaultReranker, NoopReranker } from './reranker.js';
export {
  OpenAIEmbeddingProvider,
  NoopEmbeddingProvider,
  cosineSimilarity,
  l2Normalize,
} from './embedding-provider.js';
export type { OpenAIEmbeddingProviderConfig } from './embedding-provider.js';
export { VectorIndex } from './vector-index.js';
export type { VectorEntry, VectorSearchResult } from './vector-index.js';
