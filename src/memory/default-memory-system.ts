/**
 * DefaultMemorySystem — 将短期 + 长期记忆整合为完整的记忆系统
 *
 * 设计原则：
 * - 短期记忆（SlidingWindowMemory）：最近的对话上下文，用于 LLM 的 prompt 构建
 * - 长期记忆（InMemoryLongTermMemory）：持久化的事实和偏好，检索增强
 * - consolidation() 桥接两层：从短期记忆中提取关键信息，存入长期记忆
 * - consolidate() 是显式调用的，不在每次 add() 时自动运行（避免性能问题）
 * - 后续系列可以实现 PersistentMemorySystem（Redis/MySQL 后端）
 */

import type {
  ShortTermMemory,
  LongTermMemory,
  LongTermMemoryItem,
  MemorySystem,
} from '../core/memory.js';
import { SlidingWindowMemory } from '../core/memory.js';
import type { Message } from '../core/message.js';
import type { ConsolidationStrategy } from './consolidation.js';
import { SimpleConsolidationStrategy } from './consolidation.js';
import { InMemoryLongTermMemory } from './long-term-memory.js';

export interface DefaultMemorySystemConfig {
  /** 短期记忆最大 token 数 */
  shortTermMaxTokens?: number;
  /** 合并策略 */
  consolidationStrategy?: ConsolidationStrategy;
  /** 自动合并的间隔（多少条新消息后触发），0 表示不自动合并 */
  autoConsolidateInterval?: number;
}

/**
 * 完整的记忆系统实现
 *
 * 使用方式：
 * ```typescript
 * const memory = new DefaultMemorySystem({
 *   shortTermMaxTokens: 4096,
 * });
 *
 * // 添加对话消息
 * memory.shortTerm.add(userMessage('我喜欢吃辣'));
 * memory.shortTerm.add(assistantMessage('明白了，我记住了'));
 *
 * // 将短期记忆中的重要信息提炼到长期记忆
 * await memory.consolidate();
 * ```
 */
export class DefaultMemorySystem implements MemorySystem {
  readonly shortTerm: ShortTermMemory;
  readonly longTerm: LongTermMemory;

  private consolidationStrategy: ConsolidationStrategy;
  private autoConsolidateInterval: number;
  private messagesSinceLastConsolidation = 0;

  constructor(config?: DefaultMemorySystemConfig) {
    this.shortTerm = new SlidingWindowMemory(
      config?.shortTermMaxTokens ?? 4096
    );
    this.longTerm = new InMemoryLongTermMemory();
    this.consolidationStrategy =
      config?.consolidationStrategy ?? new SimpleConsolidationStrategy();
    this.autoConsolidateInterval = config?.autoConsolidateInterval ?? 0;
  }

  /**
   * 组合检索 —— 同时查询短期和长期记忆
   *
   * 返回一个包含短期上下文和长期相关记忆的包装对象。
   * 适合在构建 LLM prompt 时使用。
   */
  async queryRelevant(query: string): Promise<{
    recentMessages: Message[];
    relevantMemories: LongTermMemoryItem[];
  }> {
    const recentMessages = this.shortTerm.getMessages();
    const relevantMemories = await this.longTerm.search({
      query,
      maxResults: 5,
      minRelevance: 0.15,
    });

    return { recentMessages, relevantMemories };
  }

  /**
   * 合并：将短期记忆中的关键信息提炼到长期记忆
   *
   * v4 更新：
   * - extract() 现在是异步调用
   * - 新增 merge 步骤：将新事实与已有长期记忆合并（由 ConsolidationStrategy 实现）
   * - 对于不支持合并的策略（如 SimpleConsolidationStrategy），merge 直接返回原始事实
   *
   * 执行流程：
   * 1. 从短期记忆获取所有消息
   * 2. 运行合并策略提取事实
   * 3. 将提取的事实与已有长期记忆合并（去重、摘要合并、矛盾检测）
   * 4. 将合并后的事实存入长期记忆
   */
  async consolidate(): Promise<void> {
    const messages = this.shortTerm.getMessages();

    if (messages.length === 0) return;

    // Phase 1: 提取事实
    const facts = await this.consolidationStrategy.extract(messages);

    if (facts.length === 0) return;

    // Phase 2: 合并（去重、矛盾检测、摘要合并）
    const existingMemories = await this.longTerm.search({
      query: '',
      maxResults: 100,
    });
    const mergedFacts = await this.consolidationStrategy.merge(
      facts,
      existingMemories,
    );

    // Phase 3: 存入长期记忆
    for (const fact of mergedFacts) {
      await this.longTerm.store({
        content: fact.content,
        metadata: {
          timestamp: Date.now(),
          type: fact.type === 'preference' ? 'user_preference'
            : fact.type === 'decision' ? 'insight'
            : 'fact',
          tags: fact.tags,
          importance: fact.importance,
        },
      });
    }

    this.messagesSinceLastConsolidation = 0;
  }

  /**
   * 遗忘 —— 根据策略清理长期记忆
   *
   * 使用复合遗忘策略：
   * - 重要性维度：删除 importance < 0.3 的
   * - 时效维度：超过 7 天半衰期，超过 90 天必忘
   * - 容量维度：超过 500 条时启用
   *
   * 可传入自定义策略覆盖默认行为。
   *
   * @param strategy 可选的自定义遗忘策略。不传则使用默认复合策略。
   * @returns 删除的条目数量
   */
  async forget(
    strategy?: import('./forgetting.js').ForgettingStrategy,
  ): Promise<number> {
    const { ForgetExecutor, CompositeForgetting, ImportanceForgetting, AgeBasedForgetting, CapacityForgetting } = await import('./forgetting.js');

    const defaultStrategy = new CompositeForgetting([
      { strategy: new ImportanceForgetting(0.3), weight: 0.5 },
      {
        strategy: new AgeBasedForgetting(
          7 * 24 * 60 * 60 * 1000, // 7 天半衰期
          90 * 24 * 60 * 60 * 1000, // 90 天最大年龄
        ),
        weight: 0.3,
      },
      {
        strategy: new CapacityForgetting(500),
        weight: 0.2,
      },
    ]);

    const executor = new ForgetExecutor({
      strategy: strategy ?? defaultStrategy,
      scoreThreshold: 0.6,
      maxDeletionsPerRun: 50,
    });

    const result = await executor.execute(
      this.longTerm as any,
    );
    return result.deletedCount;
  }

  /**
   * 添加消息并触发自动合并（如果配置了 autoConsolidateInterval）
   */
  addAndMaybeConsolidate(msg: Message): void {
    this.shortTerm.add(msg);
    this.messagesSinceLastConsolidation++;

    if (
      this.autoConsolidateInterval > 0 &&
      this.messagesSinceLastConsolidation >= this.autoConsolidateInterval
    ) {
      // 异步执行，不阻塞
      this.consolidate().catch((err) => {
        console.error('Auto-consolidation failed:', err);
      });
    }
  }
}
