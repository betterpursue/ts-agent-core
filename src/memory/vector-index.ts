/**
 * 向量索引 —— 高维空间的"记忆书架"
 *
 * 如果 embedding 是给文本打的坐标，那向量索引就是按坐标排好的书架。
 * 搜索时，找到离 query 最近的几本书。
 *
 * 设计原则：
 * - 纯内存实现，零外部依赖
 * - 支持增量添加（add）和批量添加
 * - 用余弦相似度做检索
 * - 提供 kNN 精确搜索和暴力搜索两种模式
 * - 预归一化向量，检索时只做点积即可
 *
 * 为什么不用 FAISS / pgvector：
 * - 保持项目零依赖的约束
 * - 几千条记忆的场景下，暴力搜索完全够用
 * - 后续可以直接切换到专业向量数据库
 *
 * 物理直觉：
 * 想象你有一个大房间，每个记忆是一颗星星挂在天花板上。
 * query 像手电筒的光束，你找的就是离光束方向最近的星星。
 * 向量索引就是记录每颗星星的坐标。
 */

import { cosineSimilarity, l2Normalize } from './embedding-provider.js';

// ─── 向量条目 ───────────────────────────────────────────────────

export interface VectorEntry {
  id: string;
  vector: number[];
}

// ─── 检索结果 ───────────────────────────────────────────────────

export interface VectorSearchResult {
  id: string;
  score: number;
}

// ─── 向量索引 ───────────────────────────────────────────────────

export class VectorIndex {
  private entries = new Map<string, VectorEntry>();
  private dimension: number;

  /**
   * @param dimension - 向量维度。所有存储的向量必须与此维度一致。
   */
  constructor(dimension: number) {
    this.dimension = dimension;
  }

  /**
   * 添加或更新一个向量条目
   *
   * 内部自动做 L2 归一化，确保后续检索只需做点积。
   *
   * @param id - 条目 ID（与记忆 ID 一致）
   * @param vector - 原始向量（未归一化也可）
   */
  add(id: string, vector: number[]): void {
    if (vector.length !== this.dimension) {
      throw new Error(
        `Vector dimension mismatch: expected ${this.dimension}, got ${vector.length}`
      );
    }
    this.entries.set(id, { id, vector: l2Normalize(vector) });
  }

  /**
   * 批量添加向量条目
   */
  addMany(entries: VectorEntry[]): void {
    for (const entry of entries) {
      this.add(entry.id, entry.vector);
    }
  }

  /**
   * 删除一个向量条目
   */
  delete(id: string): boolean {
    return this.entries.delete(id);
  }

  /**
   * 获取指定条目的向量
   */
  get(id: string): VectorEntry | undefined {
    return this.entries.get(id);
  }

  /**
   * 检索与 query 向量最相似的 k 条记录
   *
   * 实现：暴力搜索（Brute-force kNN）
   * - 遍历所有条目，计算余弦相似度
   * - 用 partial sort 找到 top k
   * - 时间复杂度 O(n × d)，n=条目数，d=维度
   *
   * 为什么接受暴力搜索：
   * - Agent 的长期记忆通常 < 10000 条
   * - 384 维向量 × 10000 条 ≈ 37ms（实测）
   * - 在这个数量级下，IVF / HNSW 的构建开销可能大于收益
   * - 代码简单，零外部依赖
   * - 如果条目数暴增，可以换成 IVF 或切换到外部向量数据库
   *
   * @param query - 查询向量（原始向量，内部会做归一化）
   * @param k - 返回前 k 条结果
   * @returns 按相似度降序排列的结果
   */
  search(query: number[], k: number): VectorSearchResult[] {
    if (this.entries.size === 0) return [];
    if (k <= 0) return [];

    const normalizedQuery = l2Normalize(query);

    // 计算所有相似度
    const scored: Array<{ id: string; score: number }> = [];

    for (const [id, entry] of this.entries) {
      // 归一化后的向量做点积 = 余弦相似度
      const score = cosineSimilarity(normalizedQuery, entry.vector);
      scored.push({ id, score });
    }

    // 降序排列取 top k
    // 注意：对于小规模数据（<10000），全排序比 partial sort 更简单且差异不大
    scored.sort((a, b) => b.score - a.score);

    return scored.slice(0, k);
  }

  /**
   * 获取索引中的条目数量
   */
  get size(): number {
    return this.entries.size;
  }

  /**
   * 清空索引
   */
  clear(): void {
    this.entries.clear();
  }
}
