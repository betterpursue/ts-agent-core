/**
 * 长期记忆 —— 内存实现
 *
 * 支持两种检索模式：
 * 1. 关键词检索（默认）：TF-IDF 风格的重叠词评分，零外部依赖
 * 2. 语义检索（可选）：基于 Embedding 的余弦相似度搜索，需配置 EmbeddingProvider
 *
 * 设计原则：
 * - 零外部依赖，纯 TypeScript 实现
 * - EmbeddingProvider 可替换，不影响核心逻辑
 * - 两种检索模式通过 search 接口对外统一
 * - 后续系列可以实现 RedisLongTermMemory / PGVectorMemory 替换此实现
 */

import type {
  EmbeddingProvider,
  LongTermMemoryItem,
  LongTermMemoryQuery,
  LongTermMemory,
} from '../core/memory.js';
import { VectorIndex } from './vector-index.js';

/** 默认最小相关度阈值 */
const DEFAULT_MIN_RELEVANCE = 0.1;

/** 默认最大返回结果数 */
const DEFAULT_MAX_RESULTS = 10;

/** 默认语义检索相关度阈值 */
const DEFAULT_SEMANTIC_MIN_RELEVANCE = 0.3;

/**
 * 搜索模式 —— 控制 InMemoryLongTermMemory 使用哪种检索策略
 */
export type SearchMode = 'keyword' | 'semantic' | 'hybrid';

/**
 * 内存中的长期记忆存储
 *
 * 核心逻辑：
 * - 所有数据存在 Map 里，按 id 索引
 * - 支持两种检索模式：关键词（默认）和语义（需配置 embeddingProvider）
 * - 支持 tag 过滤和时间范围过滤
 *
 * 使用方式（纯关键词检索，零依赖）：
 * ```typescript
 * const mem = new InMemoryLongTermMemory();
 * ```
 *
 * 使用方式（语义检索，需 API key）：
 * ```typescript
 * const embedder = new OpenAIEmbeddingProvider();
 * const mem = new InMemoryLongTermMemory({
 *   embeddingProvider: embedder,
 *   searchMode: 'semantic',
 * });
 * ```
 */
export class InMemoryLongTermMemory implements LongTermMemory {
  private items = new Map<string, LongTermMemoryItem>();
  private embeddingProvider?: EmbeddingProvider;
  private vectorIndex?: VectorIndex;
  private searchMode: SearchMode;

  /**
   * @param options.embeddingProvider - 可选的 Embedding Provider。
   *        配置后，store() 会自动生成 embedding，search() 可以使用语义检索。
   * @param options.searchMode - 检索模式。
   *        'keyword'（默认）| 'semantic' | 'hybrid'
   *        仅当提供了 embeddingProvider 时，semantic 和 hybrid 才生效。
   * @param options.embeddingDimension - 向量维度（默认 384）。
   *        必须与 EmbeddingProvider 的 dimensions 一致。
   */
  constructor(options?: {
    embeddingProvider?: EmbeddingProvider;
    searchMode?: SearchMode;
    embeddingDimension?: number;
  }) {
    this.embeddingProvider = options?.embeddingProvider;
    this.searchMode = options?.searchMode ?? 'keyword';
    if (this.embeddingProvider) {
      this.vectorIndex = new VectorIndex(
        options?.embeddingDimension ?? 384
      );
    }
  }

  async store(
    item: Omit<LongTermMemoryItem, 'id'>
  ): Promise<string> {
    const id = this.generateId();
    const fullItem: LongTermMemoryItem = {
      ...item,
      id,
      metadata: {
        ...item.metadata,
        timestamp: item.metadata?.timestamp ?? Date.now(),
      },
    };

    // 如果配置了 embeddingProvider，自动生成并存储 embedding
    if (this.embeddingProvider && this.vectorIndex) {
      try {
        const embedding = await this.embeddingProvider.embed(item.content);
        fullItem.embedding = embedding;
        this.vectorIndex.add(id, embedding);
      } catch (err) {
        // embedding 生成失败不阻断 store 流程
        // 后续 search 遇到无 embedding 的条目会回退到关键词匹配
        console.warn(
          `[InMemoryLongTermMemory] Failed to generate embedding for item ${id}:`,
          err instanceof Error ? err.message : String(err)
        );
      }
    }

    this.items.set(id, fullItem);
    return id;
  }

  async search(
    query: LongTermMemoryQuery
  ): Promise<LongTermMemoryItem[]> {
    const maxResults = query.maxResults ?? DEFAULT_MAX_RESULTS;

    // 如果没有查询文本，按时间倒序返回最近的 items（所有模式一致）
    if (!query.query || query.query.trim().length === 0) {
      return this.getRecentItems(maxResults, query.tagFilter);
    }

    // 根据 searchMode 选择检索策略
    switch (this.searchMode) {
      case 'semantic':
        return this.semanticSearch(query, maxResults);
      case 'hybrid':
        return this.hybridSearch(query, maxResults);
      case 'keyword':
      default:
        return this.keywordSearch(query, maxResults);
    }
  }

  /**
   * 关键词检索 —— TF-IDF 风格的重叠词评分（默认模式）
   *
   * 与 v1 保持一致，零外部依赖。
   */
  private keywordSearch(
    query: LongTermMemoryQuery,
    maxResults: number
  ): Promise<LongTermMemoryItem[]> {
    const minRelevance = query.minRelevance ?? DEFAULT_MIN_RELEVANCE;
    const queryTokens = this.tokenize(query.query!);

    if (queryTokens.length === 0) {
      return Promise.resolve(
        this.getRecentItems(maxResults, query.tagFilter)
      );
    }

    const scored: Array<{ item: LongTermMemoryItem; score: number }> = [];

    for (const item of this.items.values()) {
      if (query.tagFilter && !this.matchesTags(item, query.tagFilter)) {
        continue;
      }

      const itemTokens = this.tokenize(item.content);
      const score = this.computeRelevance(queryTokens, itemTokens);

      if (score >= minRelevance) {
        scored.push({ item, score });
      }
    }

    scored.sort((a, b) => b.score - a.score);
    return Promise.resolve(scored.slice(0, maxResults).map((s) => s.item));
  }

  /**
   * 语义检索 —— 基于 Embedding 的余弦相似度搜索
   *
   * 把 query 也转为向量，然后在向量索引中找最近邻。
   * 能捕获同义词和 paraphrase 层面的语义关联。
   *
   * 兜底逻辑：
   * - 如果没有配置 embedding provider，回退到关键词检索
   * - 如果 query embedding 生成失败，回退到关键词检索
   * - 没有 embedding 的条目（存储时 embedding 失败）不会被检索到
   */
  private async semanticSearch(
    query: LongTermMemoryQuery,
    maxResults: number
  ): Promise<LongTermMemoryItem[]> {
    if (!this.embeddingProvider || !this.vectorIndex) {
      return this.keywordSearch(query, maxResults);
    }

    if (this.vectorIndex.size === 0) {
      return this.keywordSearch(query, maxResults);
    }

    try {
      const queryEmbedding = await this.embeddingProvider.embed(
        query.query!
      );
      const vectorResults = this.vectorIndex.search(queryEmbedding, maxResults);

      const results: LongTermMemoryItem[] = [];
      for (const vr of vectorResults) {
        const item = this.items.get(vr.id);
        if (!item) continue;
        if (query.tagFilter && !this.matchesTags(item, query.tagFilter)) {
          continue;
        }
        results.push(item);
      }

      return results;
    } catch (err) {
      console.warn(
        '[InMemoryLongTermMemory] Semantic search failed, falling back to keyword:',
        err instanceof Error ? err.message : String(err)
      );
      return this.keywordSearch(query, maxResults);
    }
  }

  /**
   * 混合检索 —— RRF 融合关键词和语义检索结果
   *
   * Reciprocal Rank Fusion:
   * score(id) = Σ 1 / (k + rank_i)
   * k = 60（标准常数）
   *
   * 融合两个排序列表，不需要分数归一化。
   */
  private async hybridSearch(
    query: LongTermMemoryQuery,
    maxResults: number
  ): Promise<LongTermMemoryItem[]> {
    const keywordResults = await this.keywordSearch(query, maxResults * 2);

    let semanticResults: LongTermMemoryItem[] = [];
    if (this.embeddingProvider && this.vectorIndex && this.vectorIndex.size > 0) {
      try {
        semanticResults = await this.semanticSearch(query, maxResults * 2);
      } catch {
        // fall through: semantic failed, use keyword only
      }
    }

    if (semanticResults.length === 0) {
      return keywordResults.slice(0, maxResults);
    }

    const RRF_K = 60;
    const scoreMap = new Map<string, number>();

    for (let i = 0; i < keywordResults.length; i++) {
      const current = scoreMap.get(keywordResults[i].id) ?? 0;
      scoreMap.set(keywordResults[i].id, current + 1 / (RRF_K + i + 1));
    }

    for (let i = 0; i < semanticResults.length; i++) {
      const current = scoreMap.get(semanticResults[i].id) ?? 0;
      scoreMap.set(semanticResults[i].id, current + 1 / (RRF_K + i + 1));
    }

    const sorted = Array.from(scoreMap.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, maxResults);

    return sorted
      .map(([id]) => this.items.get(id))
      .filter((item): item is LongTermMemoryItem => item !== null);
  }

  async get(id: string): Promise<LongTermMemoryItem | null> {
    return this.items.get(id) ?? null;
  }

  async delete(id: string): Promise<boolean> {
    const deleted = this.items.delete(id);
    if (deleted && this.vectorIndex) {
      this.vectorIndex.delete(id);
    }
    return deleted;
  }

  /**
   * 遗忘 —— 移除低于重要性阈值的记忆条目
   *
   * 遍历所有条目，删除 importance < threshold 的。
   * 不设置 importance 的条目（undefined）默认保留。
   *
   * @returns 被删除的条目数量
   */
  async forget(threshold: number): Promise<number> {
    let removed = 0;
    for (const [id, item] of this.items) {
      const importance = item.metadata.importance;
      if (importance !== undefined && importance < threshold) {
        this.items.delete(id);
        this.vectorIndex?.delete(id);
        removed++;
      }
    }
    return removed;
  }

  /**
   * 获取所有条目数量（方便测试和监控）
   */
  get size(): number {
    return this.items.size;
  }

  /**
   * 清空所有条目
   */
  clear(): void {
    this.items.clear();
    this.vectorIndex?.clear();
  }

  /**
   * 生成唯一 ID
   */
  private generateId(): string {
    return `mem_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
  }

  /**
   * 获取最近的 items（无查询时的兜底方案）
   */
  private getRecentItems(
    maxResults: number,
    tagFilter?: string[]
  ): LongTermMemoryItem[] {
    const candidates = Array.from(this.items.values())
      .filter((item) => !tagFilter || this.matchesTags(item, tagFilter))
      .sort((a, b) => b.metadata.timestamp - a.metadata.timestamp);

    return candidates.slice(0, maxResults);
  }

  /**
   * 检查 item 是否匹配所有指定 tag
   */
  private matchesTags(item: LongTermMemoryItem, tags: string[]): boolean {
    if (!item.metadata.tags || item.metadata.tags.length === 0) {
      return false;
    }
    return tags.every((t) => item.metadata.tags!.includes(t));
  }

  /**
   * 将文本拆成小写关键词
   *
   * 中文：逐字符拆（中文字符本身就是语义单元）
   * 英文：按空格和标点拆分
   */
  private tokenize(text: string): string[] {
    const tokens: string[] = [];
    // 提取英文单词
    const enWords = text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
    tokens.push(...enWords);
    // 提取中文（逐字符）
    const cnChars = text.match(/[\u4e00-\u9fff]/g) ?? [];
    tokens.push(...cnChars);
    return tokens;
  }

  /**
   * 计算查询和文档之间的相关度
   *
   * 算法：简单重叠评分
   * - 对每个查询 token，看它在文档 token 中出现的频率
   * - 出现频率越高，分数越高
   * - 除以文档长度做归一化
   * - 用 log 平滑频次
   */
  private computeRelevance(
    queryTokens: string[],
    itemTokens: string[]
  ): number {
    if (queryTokens.length === 0 || itemTokens.length === 0) return 0;

    // 统计查询 token 在 item 中的出现次数
    let matchScore = 0;
    for (const qt of queryTokens) {
      const count = itemTokens.filter((it) => it === qt).length;
      if (count > 0) {
        // log(1 + count) 做平滑，避免长文档中单次高频词过度主导
        matchScore += Math.log(1 + count);
      }
    }

    // 归一化：除以最大可能的分数（查询长度 * log(2)）
    const maxScore = queryTokens.length * Math.log(2);
    const normalizedScore = Math.min(matchScore / maxScore, 1.0);

    // 加上 Token 覆盖率因子——命中的查询 token 种类越多，加分越多
    const uniqueQueryTokens = new Set(queryTokens);
    const uniqueItemTokens = new Set(itemTokens);
    let coveredCount = 0;
    for (const qt of uniqueQueryTokens) {
      if (uniqueItemTokens.has(qt)) coveredCount++;
    }
    const coverageFactor = coveredCount / uniqueQueryTokens.size;

    // 最终分数 = 频次分 * 0.6 + 覆盖分 * 0.4
    return Math.min(normalizedScore * 0.6 + coverageFactor * 0.4, 1.0);
  }
}
