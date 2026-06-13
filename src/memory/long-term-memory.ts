/**
 * 长期记忆 —— 内存实现
 *
 * 用 TF-IDF 风格的关键词检索代替 embedding 向量搜索，
 * 避免引入外部依赖（向量数据库、Embedding 模型等）。
 *
 * 设计原则：
 * - 零外部依赖，纯 TypeScript 实现
 * - 检索不依赖 embedding，适合本地开发和测试
 * - 后续系列可以实现 RedisLongTermMemory / PGVectorMemory 替换此实现
 */

import type {
  LongTermMemoryItem,
  LongTermMemoryQuery,
  LongTermMemory,
} from '../core/memory.js';

/** 默认最小相关度阈值 */
const DEFAULT_MIN_RELEVANCE = 0.1;

/** 默认最大返回结果数 */
const DEFAULT_MAX_RESULTS = 10;

/**
 * 内存中的长期记忆存储
 *
 * 核心逻辑：
 * - 所有数据存在 Map 里，按 id 索引
 * - 检索时对 content 做关键词拆解，用 TF-IDF 风格评分
 * - 支持 tag 过滤和时间范围过滤（通过 metadata）
 */
export class InMemoryLongTermMemory implements LongTermMemory {
  private items = new Map<string, LongTermMemoryItem>();

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
    this.items.set(id, fullItem);
    return id;
  }

  async search(
    query: LongTermMemoryQuery
  ): Promise<LongTermMemoryItem[]> {
    const maxResults = query.maxResults ?? DEFAULT_MAX_RESULTS;
    const minRelevance = query.minRelevance ?? DEFAULT_MIN_RELEVANCE;

    // 如果没有查询文本，按时间倒序返回最近的 items
    if (!query.query || query.query.trim().length === 0) {
      return this.getRecentItems(maxResults, query.tagFilter);
    }

    // 对查询做关键词拆解
    const queryTokens = this.tokenize(query.query);
    if (queryTokens.length === 0) {
      return this.getRecentItems(maxResults, query.tagFilter);
    }

    // 计算每个 item 的 TF-IDF 风格相关度
    const scored: Array<{ item: LongTermMemoryItem; score: number }> = [];

    for (const item of this.items.values()) {
      // tag 过滤
      if (query.tagFilter && !this.matchesTags(item, query.tagFilter)) {
        continue;
      }

      const itemTokens = this.tokenize(item.content);
      const score = this.computeRelevance(queryTokens, itemTokens);

      if (score >= minRelevance) {
        scored.push({ item, score });
      }
    }

    // 按相关度降序排列
    scored.sort((a, b) => b.score - a.score);

    return scored.slice(0, maxResults).map((s) => s.item);
  }

  async get(id: string): Promise<LongTermMemoryItem | null> {
    return this.items.get(id) ?? null;
  }

  async delete(id: string): Promise<boolean> {
    return this.items.delete(id);
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
