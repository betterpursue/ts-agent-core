/**
 * 检索结果重排器 —— 时效加权 + MMR 多样性重排
 *
 * 设计原则：
 * - 零外部依赖，纯 TypeScript
 * - 不修改 LongTermMemory 接口，作为 MemoryInjector 的可选后处理 pipeline
 * - 时效加权和 MMR 可独立使用也可组合
 * - 组合时：先时效加权排序，再 MMR 多样性重排
 * - MMR 通过 Jaccard 相似度实现，不依赖 embedding
 *
 * Pipeline 流程：
 *   search() 原始结果
 *       ↓
 *   每条结果计算 Jaccard 相似度 + 时效衰减分数
 *       ↓
 *   若 recencyWeight > 0，组合分数并重排序
 *       ↓
 *   对 top N 候选执行 MMR 多样性重排
 *       ↓
 *   返回最终重排结果
 */

import type { LongTermMemoryItem } from '../core/memory.js';

// =============== 重排器配置与接口 ===============

/** 重排器配置 */
export interface RerankerConfig {
  /** 时效衰减权重（0-1），0=不使用时效，1=完全使用时效替代相关度 */
  recencyWeight?: number;
  /** MMR 多样性/相关度平衡参数 λ（0-1），0=纯多样性，1=纯相关度 */
  mmrLambda?: number;
  /** 时效衰减半衰期（毫秒），默认 7 天（7 * 24 * 60 * 60 * 1000） */
  recencyHalfLife?: number;
  /** MMR 候选窗口大小，从原始结果的 top N 中选（默认 20） */
  mmrCandidateWindow?: number;
  /** 最终返回条数（默认 5） */
  maxResults?: number;
}

/**
 * Reranker 相关的默认常量
 * 同时被 memory-injector.ts 引用，用来调整 search maxResults
 */
export const RERANKER_DEFAULTS = {
  recencyWeight: 0.1,
  mmrLambda: 0.7,
  recencyHalfLife: 7 * 24 * 60 * 60 * 1000, // 7 天
  mmrCandidateWindow: 20,
  maxResults: 5,
} as const;

/** 重排器接口 */
export interface Reranker {
  /**
   * 对检索结果重新排序
   *
   * @param query - 原始检索 query
   * @param results - 从 LongTermMemory.search() 获取的原始结果
   * @param config - 重排配置，覆盖默认值
   * @returns 重排后的结果列表
   */
  rerank(
    query: string,
    results: LongTermMemoryItem[],
    config?: RerankerConfig
  ): Promise<LongTermMemoryItem[]>;
}

// =============== 默认配置常量 ===============

const DEFAULT_RERANKER_CONFIG: Required<
  RerankerConfig & { recencyHalfLife: number; mmrCandidateWindow: number }
> = {
  recencyWeight: RERANKER_DEFAULTS.recencyWeight,
  mmrLambda: RERANKER_DEFAULTS.mmrLambda,
  recencyHalfLife: RERANKER_DEFAULTS.recencyHalfLife,
  mmrCandidateWindow: RERANKER_DEFAULTS.mmrCandidateWindow,
  maxResults: RERANKER_DEFAULTS.maxResults,
};

// =============== token 化 & 相似度 ===============

/**
 * 将文本拆成 token 序列
 *
 * 中文逐字拆分，英文按空格和标点拆分后取字母数字词。
 * 和 InMemoryLongTermMemory 的 tokenize 保持一致。
 */
function tokenize(text: string): string[] {
  const tokens: string[] = [];
  const enWords = text.toLowerCase().match(/[a-z0-9]+/g) ?? [];
  tokens.push(...enWords);
  const cnChars = text.match(/[\u4e00-\u9fff]/g) ?? [];
  tokens.push(...cnChars);
  return tokens;
}

/**
 * Jaccard 相似度 —— 基于 token 集合的交集/并集
 *
 * J(A, B) = |A ∩ B| / |A ∪ B|
 *
 * 为什么用 Jaccard 而非余弦相似度：
 * - 不依赖向量长度归一化，短文本更公平
 * - 和 InMemoryLongTermMemory 的内部评分逻辑一致
 * - 可解释性强：交集/并集可以直接检查
 * - 计算快，O(|A| + |B|) 级别
 */
function jaccardSimilarity(a: string, b: string): number {
  const tokensA = tokenize(a);
  const tokensB = tokenize(b);

  if (tokensA.length === 0 && tokensB.length === 0) return 1;
  if (tokensA.length === 0 || tokensB.length === 0) return 0;

  const setA = new Set(tokensA);
  const setB = new Set(tokensB);

  let intersection = 0;
  for (const t of setA) {
    if (setB.has(t)) intersection++;
  }

  const union = setA.size + setB.size - intersection;
  return union === 0 ? 0 : intersection / union;
}

// =============== 时效衰减 ===============

/**
 * 指数衰减函数
 *
 * 用物理学中的半衰期概念建模遗忘曲线：
 *   score = e^(-t / τ)
 *   其中 τ = halfLife / ln(2)
 *
 * 核心特性：
 * - 当 t = 0 时，score = 1（刚刚发生的，完全记住）
 * - 当 t = halfLife 时，score = 0.5（半衰期，重要性减半）
 * - 当 t → ∞ 时，score → 0（无限久远，完全遗忘）
 *
 * 为什么是指数衰减而非线性或对数：
 * - 最接近 Ebbinghaus 遗忘曲线
 * - 平滑连续，无断点
 * - 单一参数（半衰期），物理意义明确
 * - 衰减速率随时间变化（先快后慢），符合直觉
 */
function exponentialDecay(timestamp: number, halfLife: number): number {
  const elapsed = Date.now() - timestamp;
  if (elapsed <= 0) return 1;

  // τ = halfLife / ln(2)，保证当 elapsed = halfLife 时结果为 0.5
  const tau = halfLife / Math.LN2;
  return Math.exp(-elapsed / tau);
}

// =============== 内部数据结构 ===============

/**
 * 带评分的记忆条目（内部使用）
 */
interface ScoredItem {
  item: LongTermMemoryItem;
  /** 与检索 query 的 Jaccard 相似度 */
  querySimilarity: number;
  /** 时效衰减分数（0-1） */
  recencyScore: number;
  /** 与其他候选条目的 pairwise Jaccard 相似度缓存 */
  pairwiseSim: Map<string, number>;
}

// =============== MMR 核心算法 ===============

/**
 * Maximal Marginal Relevance（最大边界相关度）
 *
 * 从候选集中选出既与 query 相关、又和已选结果不重复的子集。
 *
 * 算法伪代码：
 * ```
 * S = []           // 已选集合
 * C = candidates   // 候选集
 *
 * while |S| < k and C ≠ ∅:
 *   d* = argmax_{d ∈ C} [ λ * sim(d, q) - (1-λ) * max_{s ∈ S} sim(d, s) ]
 *   S.push(d*)
 *   C.remove(d*)
 *
 * return S
 * ```
 *
 * 第一项 λ * sim(d, q) 鼓励选与 query 相关的结果。
 * 第二项 (1-λ) * max(sim(d, s)) 惩罚与已选结果过于相似的结果。
 * λ 控制两边的权重：
 *   - λ = 1：纯相关度排序（和原始 search 一致）
 *   - λ = 0：纯多样性（不考虑 query，只互相不相似）
 *   - λ = 0.7：推荐值，偏相关度但保留多样性
 *
 * @param candidates - 候选列表（已按相关度排序）
 * @param lambda - MMR λ 参数
 * @param numToSelect - 要选出的结果数
 * @returns 按 MMR 分数降序排列的结果
 */
function mmrSelect(
  candidates: ScoredItem[],
  lambda: number,
  numToSelect: number
): ScoredItem[] {
  if (candidates.length === 0) return [];
  if (numToSelect <= 0) return [];

  const selected: ScoredItem[] = [];
  const remaining = [...candidates];
  const targetCount = Math.min(numToSelect, candidates.length);

  while (selected.length < targetCount && remaining.length > 0) {
    let bestIdx = 0;
    let bestScore = -Infinity;

    for (let i = 0; i < remaining.length; i++) {
      const score = computeMMRScore(remaining[i], selected, lambda);
      if (score > bestScore) {
        bestScore = score;
        bestIdx = i;
      }
    }

    selected.push(remaining[bestIdx]);
    remaining.splice(bestIdx, 1);
  }

  return selected;
}

/**
 * 计算单条候选的 MMR 分数
 *
 * MMR(d) = λ * sim(d, q) - (1-λ) * max_{s ∈ S} sim(d, s)
 */
function computeMMRScore(
  item: ScoredItem,
  selected: ScoredItem[],
  lambda: number
): number {
  const relevanceTerm = lambda * item.querySimilarity;

  let maxSimToSelected = 0;
  for (const sel of selected) {
    const sim = item.pairwiseSim.get(sel.item.id) ?? 0;
    if (sim > maxSimToSelected) maxSimToSelected = sim;
  }

  return relevanceTerm - (1 - lambda) * maxSimToSelected;
}

// =============== 默认重排器实现 ===============

/**
 * DefaultReranker —— 集时效加权和 MMR 多样性重排于一体的默认重排器
 *
 * Pipeline 详细说明：
 *
 * 1. 计算 Jaccard 相似度
 *    对每条原始结果，计算其 content 与 query 的 Jaccard 相似度。
 *    结果范围 [0, 1]，值越大表示语义越相关。
 *
 * 2. 计算时效衰减分数
 *    根据每条结果的 metadata.timestamp，计算指数衰减分数。
 *    结果范围 [0, 1]，越近的分数越高。
 *
 * 3. 时效加权（当 recencyWeight > 0 时）
 *    组合公式：blendedSim = jaccard * (1 - w) + recency * w
 *    这里 w = recencyWeight（默认 0.1）
 *    w = 0.1 意味着时效贡献 10%，相关度贡献 90%——微调而非颠覆
 *    按 blendedSim 降序排列，确定 MMR 候选顺序
 *
 * 4. 截取候选窗口
 *    取 top mmrCandidateWindow 条作为 MMR 的候选池。
 *    默认 20 条，因为 MMR 是 O(k * n²) 的，太大会影响性能。
 *
 * 5. 预计算 pairwise 相似度
 *    计算候选池中每对记忆之间的 Jaccard 相似度，缓存到 pairwiseSim。
 *    这是 MMR 多样性惩罚的基础。
 *
 * 6. MMR 主循环
 *    从候选池中逐条选出 MMR 分数最高的结果，加入最终结果集。
 *    直到选出 maxResults 条或候选池耗尽。
 *
 * 使用示例：
 * ```typescript
 * const reranker = new DefaultReranker({
 *   recencyWeight: 0.15,
 *   mmrLambda: 0.7,
 *   recencyHalfLife: 3 * 24 * 60 * 60 * 1000, // 3 天
 * });
 *
 * const results = await memory.search({ query, maxResults: 20 });
 * const reranked = await reranker.rerank(query, results, {
 *   maxResults: 5,
 * });
 * ```
 */
export class DefaultReranker implements Reranker {
  private defaultConfig: Required<
    RerankerConfig & { recencyHalfLife: number; mmrCandidateWindow: number }
  >;

  constructor(config?: Partial<RerankerConfig>) {
    this.defaultConfig = { ...DEFAULT_RERANKER_CONFIG, ...config } as Required<
      RerankerConfig & { recencyHalfLife: number; mmrCandidateWindow: number }
    >;
  }

  async rerank(
    query: string,
    results: LongTermMemoryItem[],
    config?: RerankerConfig
  ): Promise<LongTermMemoryItem[]> {
    // 边界情况：空或单条结果不需要重排
    if (results.length <= 1) return results;

    const cfg = {
      ...this.defaultConfig,
      ...config,
    } as Required<
      RerankerConfig & { recencyHalfLife: number; mmrCandidateWindow: number }
    >;

    // Step 1: 计算每项的 Jaccard 相似度和时效分数
    const scoredItems = results.map((item) => ({
      item,
      querySimilarity: jaccardSimilarity(query, item.content),
      recencyScore: exponentialDecay(
        item.metadata.timestamp,
        cfg.recencyHalfLife
      ),
      pairwiseSim: new Map<string, number>(),
    }));

    // Step 2: 时效加权 —— 将 recency 融入 querySimilarity
    if (cfg.recencyWeight > 0) {
      const w = cfg.recencyWeight;
      for (const si of scoredItems) {
        si.querySimilarity =
          si.querySimilarity * (1 - w) + si.recencyScore * w;
      }
    }

    // Step 3: 按（加权后的）相关度排序，取 top N 作为 MMR 候选
    scoredItems.sort((a, b) => b.querySimilarity - a.querySimilarity);
    const candidateWindow = scoredItems.slice(0, cfg.mmrCandidateWindow);

    // Step 4: 预计算候选之间的 pairwise Jaccard 相似度
    for (let i = 0; i < candidateWindow.length; i++) {
      for (let j = i + 1; j < candidateWindow.length; j++) {
        const sim = jaccardSimilarity(
          candidateWindow[i].item.content,
          candidateWindow[j].item.content
        );
        candidateWindow[i].pairwiseSim.set(
          candidateWindow[j].item.id,
          sim
        );
        candidateWindow[j].pairwiseSim.set(
          candidateWindow[i].item.id,
          sim
        );
      }
    }

    // Step 5: MMR 多样性重排
    const mmrSelected = mmrSelect(
      candidateWindow,
      cfg.mmrLambda,
      cfg.maxResults
    );

    return mmrSelected.map((si) => si.item);
  }
}

/**
 * NoopReranker —— 不做任何重排的空实现
 *
 * 遵循 Null Object Pattern，用于"关闭重排"的场景。
 * 测试时也可以用它作为基准对照组。
 */
export class NoopReranker implements Reranker {
  async rerank(
    _query: string,
    results: LongTermMemoryItem[],
    _config?: RerankerConfig
  ): Promise<LongTermMemoryItem[]> {
    return results;
  }
}
