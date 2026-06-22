/**
 * 遗忘机制（Forgetting）—— 让 Agent 的记忆有进有退
 *
 * 设计原则：
 * - 遗忘不是简单的"删除"，而是有策略的"清理"
 * - 支持多种遗忘策略，可组合
 * - 遗忘过程异步执行，不阻塞主流程
 * - 与重要性、时效、容量等维度联动
 *
 * 类比：
 * 人脑的记忆不是永久存储的。艾宾浩斯遗忘曲线告诉我们，
 * 没有复习的信息会随时间指数衰减。Agent 的记忆系统也需要类似的机制——
 * 否则长期记忆会无限膨胀，检索质量下降，上下文 token 被噪声占据。
 */

import type { LongTermMemoryItem, LongTermMemory } from '../core/memory.js';

// ─── 遗忘策略接口 ───────────────────────────────────────────────

/**
 * 单条记忆的"可遗忘分数"
 * 分数越高，越应该被遗忘。
 */
export interface ForgettingScore {
  /** 记忆 ID */
  itemId: string;
  /** 可遗忘分数 (0-1)，越高越容易被遗忘 */
  score: number;
  /** 遗忘原因（用于日志和调试） */
  reason: string;
  /** 产生该分数的策略名称 */
  strategyName: string;
}

/**
 * 遗忘策略 —— 为每条记忆计算一个"可遗忘分数"
 *
 * 不同的策略从不同维度评估：
 * - 重要性维度：低重要性 = 容易被遗忘
 * - 时效维度：越久远 = 越容易被遗忘
 * - 容量维度：同类型中排序靠后的 = 容易被遗忘
 */
export interface ForgettingStrategy {
  /** 策略名称（用于日志和配置） */
  readonly name: string;

  /**
   * 为所有记忆计算可遗忘分数
   *
   * @param items    当前所有长期记忆
   * @param context  额外上下文（如当前容量、时间等）
   * @returns        每条记忆的可遗忘分数
   */
  score(
    items: LongTermMemoryItem[],
    context: ForgettingContext,
  ): Promise<ForgettingScore[]>;
}

/**
 * 遗忘上下文 —— 策略计算时需要的环境信息
 */
export interface ForgettingContext {
  /** 当前时间戳（ms） */
  now: number;
  /** 记忆存储的容量上限（条数），0 表示无上限 */
  capacity: number;
  /** 当前记忆总数 */
  currentSize: number;
}

// ─── 内置遗忘策略 ───────────────────────────────────────────────

/**
 * 基于重要性的遗忘
 *
 * 删除重要性低于阈值的记忆。
 * 这是最直接的遗忘方式：LLM 认为不重要的信息，直接清理。
 *
 * 公式：
 *   score = 1 - importance
 *   （重要性越低，可遗忘分数越高）
 */
export class ImportanceForgetting implements ForgettingStrategy {
  readonly name = 'importance';
  private threshold: number;

  constructor(threshold: number = 0.3) {
    this.threshold = threshold;
  }

  async score(items: LongTermMemoryItem[]): Promise<ForgettingScore[]> {
    return items.map((item) => {
      const importance = item.metadata.importance ?? 0.5;
      const score = Math.max(0, 1 - importance);
      const shouldForget = importance < this.threshold;
      return {
        itemId: item.id,
        score,
        reason: shouldForget
          ? `importance ${importance.toFixed(2)} < threshold ${this.threshold}`
          : `importance ${importance.toFixed(2)} >= threshold`,
        strategyName: this.name,
      };
    });
  }
}

/**
 * 基于时效的遗忘
 *
 * 超过指定年龄的记忆，随时间推移可遗忘分数逐渐升高。
 * 使用指数衰减的反向：记忆越老，衰减越严重。
 *
 * 公式：
 *   age = now - timestamp
 *   recency = exp(-age / (halfLife / ln(2)))
 *   score = 1 - recency  （recency 越低，score 越高）
 *
 * 半衰期参数让调参有物理意义：
 * - halfLife = 7天：7 天前的记忆新鲜度减半，score = 0.5
 * - halfLife = 30天：一个月前的记忆新鲜度减半
 */
export class AgeBasedForgetting implements ForgettingStrategy {
  readonly name = 'age_based';
  private halfLife: number;
  private maxAge: number;

  /**
   * @param halfLife 半衰期（毫秒）。超过这个时间，记忆的"新鲜度"衰减到 0.5
   * @param maxAge   最大年龄（毫秒）。超过这个时间的记忆 score = 1（必忘）
   */
  constructor(halfLife: number, maxAge: number) {
    this.halfLife = halfLife;
    this.maxAge = maxAge;
  }

  async score(items: LongTermMemoryItem[], context: ForgettingContext): Promise<ForgettingScore[]> {
    const now = context.now;

    return items.map((item) => {
      const age = now - item.metadata.timestamp;

      // 超过 maxAge，直接标记为必忘
      if (age >= this.maxAge) {
        return {
          itemId: item.id,
          score: 1.0,
          reason: `age ${this.formatAge(age)} >= maxAge ${this.formatAge(this.maxAge)}`,
          strategyName: this.name,
        };
      }

      // 计算新鲜度（指数衰减）
      const tau = this.halfLife / Math.LN2;
      const recency = Math.exp(-age / tau);
      const score = 1 - recency;

      return {
        itemId: item.id,
        score,
        reason: `age ${this.formatAge(age)}, recency ${recency.toFixed(3)}`,
        strategyName: this.name,
      };
    });
  }

  private formatAge(ms: number): string {
    const days = ms / (1000 * 60 * 60 * 24);
    if (days >= 1) return `${days.toFixed(1)}天`;
    const hours = ms / (1000 * 60 * 60);
    if (hours >= 1) return `${hours.toFixed(1)}小时`;
    return `${ms / 60000}分钟`;
  }
}

/**
 * 基于容量的遗忘
 *
 * 当记忆数量超过容量上限时，保留最重要的，遗忘最不重要的。
 * 类似于 LRU 但用重要性代替访问时间。
 *
 * 算法：
 * 1. 按重要性降序排列
 * 2. 超过 capacity 的部分，按排序位置计算 score
 * 3. 刚好在 capacity 边界上的 score 平滑过渡
 *
 * 公式：
 *   overflow = max(0, size - capacity)
 *   rank = 该条在重要性排序中的位置（0 = 最重要）
 *   if rank < capacity: score = 0
 *   else: score = (rank - capacity + 1) / overflow
 */
export class CapacityForgetting implements ForgettingStrategy {
  readonly name = 'capacity';
  private capacity: number;

  constructor(capacity: number) {
    this.capacity = capacity;
  }

  async score(items: LongTermMemoryItem[]): Promise<ForgettingScore[]> {
    if (this.capacity <= 0) {
      // 无上限，全部 score = 0
      return items.map((item) => ({
        itemId: item.id,
        score: 0,
        reason: 'no capacity limit',
        strategyName: this.name,
      }));
    }

    // 按重要性降序排列
    const sorted = items
      .map((item) => ({
        item,
        importance: item.metadata.importance ?? 0.5,
      }))
      .sort((a, b) => b.importance - a.importance);

    const size = items.length;
    const overflow = Math.max(0, size - this.capacity);

    if (overflow === 0) {
      return items.map((item) => ({
        itemId: item.id,
        score: 0,
        reason: `within capacity (${size}/${this.capacity})`,
        strategyName: this.name,
      }));
    }

    // 建立 ID -> 排序位置的映射
    const rankMap = new Map<string, number>();
    sorted.forEach((s, rank) => rankMap.set(s.item.id, rank));

    return items.map((item) => {
      const rank = rankMap.get(item.id) ?? 0;
      if (rank < this.capacity) {
        return {
          itemId: item.id,
          score: 0,
          reason: `rank ${rank} < capacity ${this.capacity}`,
          strategyName: this.name,
        };
      }

      // 在溢出区域，score 线性增长
      const normalizedRank = rank - this.capacity + 1;
      const score = Math.min(1, normalizedRank / overflow);

      return {
        itemId: item.id,
        score,
        reason: `rank ${rank}, overflow ${overflow}, score ${score.toFixed(2)}`,
        strategyName: this.name,
      };
    });
  }
}

/**
 * 复合遗忘策略
 *
 * 将多个策略的分数加权组合，得到最终的可遗忘分数。
 * 支持自定义权重。
 *
 * 公式：
 *   finalScore = Σ (weight_i * score_i)
 *
 * 权重归一化：如果权重和不为 1，自动归一化。
 */
export class CompositeForgetting implements ForgettingStrategy {
  readonly name = 'composite';
  private strategies: Array<{ strategy: ForgettingStrategy; weight: number }>;

  constructor(
    strategies: Array<{ strategy: ForgettingStrategy; weight: number }>,
  ) {
    this.strategies = strategies;
  }

  async score(items: LongTermMemoryItem[], context: ForgettingContext): Promise<ForgettingScore[]> {
    if (this.strategies.length === 0) {
      return items.map((item) => ({
        itemId: item.id,
        score: 0,
        reason: 'no strategies configured',
        strategyName: this.name,
      }));
    }

    // 归一化权重
    const totalWeight = this.strategies.reduce((sum, s) => sum + s.weight, 0);
    const normalized = this.strategies.map((s) => ({
      strategy: s.strategy,
      weight: totalWeight > 0 ? s.weight / totalWeight : 0,
    }));

    // 并行计算所有策略的分数
    const allScores = await Promise.all(
      normalized.map((s) => s.strategy.score(items, context)),
    );

    // 合并分数
    return items.map((item, idx) => {
      let totalScore = 0;
      const reasons: string[] = [];

      for (let i = 0; i < normalized.length; i++) {
        const s = allScores[i][idx];
        totalScore += normalized[i].weight * s.score;
        if (s.score > 0.01) {
          reasons.push(`${s.strategyName}=${s.score.toFixed(2)}`);
        }
      }

      return {
        itemId: item.id,
        score: Math.min(1, totalScore),
        reason: reasons.length > 0 ? reasons.join(', ') : 'no forgetting signal',
        strategyName: this.name,
      };
    });
  }
}

// ─── 遗忘执行器 ─────────────────────────────────────────────────

/**
 * 遗忘执行器的配置
 */
export interface ForgetExecutorConfig {
  /** 遗忘策略 */
  strategy: ForgettingStrategy;
  /** 可遗忘分数阈值，超过此值的记忆将被删除。默认 0.7 */
  scoreThreshold?: number;
  /** 单次最多删除的条数（安全阀）。默认 100 */
  maxDeletionsPerRun?: number;
  /** 是否启用 dry-run 模式（只计算不删除）。默认 false */
  dryRun?: boolean;
}

/**
 * 遗忘执行结果
 */
export interface ForgetResult {
  /** 被删除的记忆数量 */
  deletedCount: number;
  /** 被保留的记忆数量 */
  keptCount: number;
  /** 每条记忆的评估详情 */
  details: Array<{
    itemId: string;
    score: number;
    deleted: boolean;
    reason: string;
  }>;
}

/**
 * 遗忘执行器 —— 在 LongTermMemory 上执行遗忘
 *
 * 职责：
 * 1. 调用策略计算所有记忆的可遗忘分数
 * 2. 根据阈值和安全阀决定删除哪些
 * 3. 返回详细结果供日志和监控使用
 */
export class ForgetExecutor {
  private config: Required<ForgetExecutorConfig>;

  constructor(config: ForgetExecutorConfig) {
    this.config = {
      strategy: config.strategy,
      scoreThreshold: config.scoreThreshold ?? 0.7,
      maxDeletionsPerRun: config.maxDeletionsPerRun ?? 100,
      dryRun: config.dryRun ?? false,
    };
  }

  /**
   * 在给定的 LongTermMemory 上执行遗忘
   *
   * 要求 memory 实现 getAllItems() 方法（InMemoryLongTermMemory 已支持）。
   */
  async execute(
    memory: LongTermMemory & {
      getAllItems(): LongTermMemoryItem[];
      delete: (id: string) => Promise<boolean>;
      size: number;
    },
  ): Promise<ForgetResult> {
    const now = Date.now();
    const items = memory.getAllItems();
    const context: ForgettingContext = {
      now,
      capacity: 0,
      currentSize: memory.size,
    };

    if (items.length === 0) {
      return { deletedCount: 0, keptCount: 0, details: [] };
    }

    // 计算可遗忘分数
    const scores = await this.config.strategy.score(items, context);

    // 决定删除哪些
    const toDelete = scores
      .filter((s) => s.score >= this.config.scoreThreshold)
      .sort((a, b) => b.score - a.score) // 分数高的先删
      .slice(0, this.config.maxDeletionsPerRun);

    const toDeleteIds = new Set(toDelete.map((s) => s.itemId));

    // 执行删除
    let deletedCount = 0;
    const details = scores.map((s) => ({
      itemId: s.itemId,
      score: s.score,
      deleted: toDeleteIds.has(s.itemId) && !this.config.dryRun,
      reason: s.reason,
    }));

    if (!this.config.dryRun) {
      for (const id of toDeleteIds) {
        await memory.delete(id);
        deletedCount++;
      }
    }

    return {
      deletedCount: this.config.dryRun ? 0 : deletedCount,
      keptCount: items.length - (this.config.dryRun ? 0 : deletedCount),
      details,
    };
  }
}
