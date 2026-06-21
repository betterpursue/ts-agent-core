/**
 * 记忆整合（Consolidation）—— 将短期记忆提炼到长期记忆
 *
 * 设计原则：
 * - ConsolidationStrategy 接口可替换，后续可以接入 LLM 做语义级摘要
 * - 默认实现基于简单规则，零外部依赖
 * - 提取的事实类型包括：用户事实、偏好、决策等
 * - 重要性评分基于频率和出现位置
 */

import type { Message } from '../core/message.js';
import type { LongTermMemoryItem } from '../core/memory.js';

/** 合并策略的配置 */
export interface ConsolidationConfig {
  /** 单次合并最多提取的事实数 */
  maxFactsPerConsolidation?: number;
  /** 重要性增益：同一事实重复出现时，每次增加多少 */
  importanceBoostPerMention?: number;
  /** 事实被看作"重要"所需的最少提及次数 */
  minMentionsForHighImportance?: number;
}

const DEFAULT_CONFIG: Required<ConsolidationConfig> = {
  maxFactsPerConsolidation: 10,
  importanceBoostPerMention: 0.15,
  minMentionsForHighImportance: 3,
};

/**
 * 从短期记忆中提取的事实
 */
export interface ExtractedFact {
  content: string;
  type: 'fact' | 'preference' | 'decision';
  tags: string[];
  importance: number;
}

/**
 * 合并策略 —— 从消息列表中提取可存储到长期记忆的事实
 *
 * v4 新增：
 * - extract() 改为异步，支持 LLM 驱动的提取
 * - merge() 用于将新事实与已有长期记忆合并（去重、矛盾检测、摘要合并）
 */
export interface ConsolidationStrategy {
  /** 从一组消息中提取事实 */
  extract(messages: Message[]): Promise<ExtractedFact[]>;

  /** 将新提取的事实与已有长期记忆合并，返回最终要存储的事实列表 */
  merge(
    newFacts: ExtractedFact[],
    existingMemories: LongTermMemoryItem[],
  ): Promise<ExtractedFact[]>;
}

/**
 * 简单的基于规则的合并策略
 *
 * 提取逻辑：
 * 1. user 消息中的关键信息（长度 > 10 且有实际内容的）
 * 2. assistant 消息中的重要回答
 * 3. 跟踪消息中反复出现的主题词，增加重要性评分
 */
export class SimpleConsolidationStrategy implements ConsolidationStrategy {
  private config: Required<ConsolidationConfig>;
  /** 话题频次统计 —— 用于跨合并周期累积重要性 */
  private mentionTracker = new Map<string, number>();

  constructor(config?: ConsolidationConfig) {
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  async extract(messages: Message[]): Promise<ExtractedFact[]> {
    const facts: ExtractedFact[] = [];

    // 从 user 消息提取
    for (const msg of messages) {
      if (msg.role === 'user') {
        const userFacts = this.extractFromUserMessage(msg);
        facts.push(...userFacts);
      }
    }

    // 对提取的事实做重要性评估
    const scored = this.scoreFacts(facts);

    // 按重要性降序，取 top N
    scored.sort((a, b) => b.importance - a.importance);

    return scored.slice(0, this.config.maxFactsPerConsolidation);
  }

  /**
   * 合并新事实与已有长期记忆 —— 默认实现不做任何合并
   *
   * LLMConsolidationStrategy 会覆盖此方法，用 LLM 分析事实之间的关系。
   */
  async merge(
    newFacts: ExtractedFact[],
    _existingMemories: LongTermMemoryItem[],
  ): Promise<ExtractedFact[]> {
    return newFacts;
  }

  /**
   * 重置话题跟踪器（通常在长时间间隔后调用）
   */
  resetTracker(): void {
    this.mentionTracker.clear();
  }

  /**
   * 从用户消息中提取事实
   *
   * 启发式规则：
   * - 太短的消息（问候、确认等）不提取
   * - 包含"我喜欢""我讨厌""我平时"等模式的消息标记为 preference
   * - 包含"决定""选择""用"等模式的消息标记为 decision
   * - 其余视为 fact
   */
  private extractFromUserMessage(msg: Message): ExtractedFact[] {
    const content = msg.content.trim();
    if (content.length < 10) {
      return []; // 太短，大概率是问候或确认
    }

    const facts: ExtractedFact[] = [];
    const type = this.classifyMessage(content);

    // 提取关键词作为 tag
    const tags = this.extractTags(content);

    facts.push({
      content,
      type,
      tags,
      importance: 0.3, // 基础重要性
    });

    return facts;
  }

  /**
   * 对事实进行重要性评分
   *
   * 评分因素：
   * - 基础重要性（由提取阶段设定）
   * - 话题提及频次加成（同一话题被提及越多越重要）
   * - 内容长度加成（更长的消息倾向于包含更多信息）
   */
  private scoreFacts(facts: ExtractedFact[]): ExtractedFact[] {
    return facts.map((fact) => {
      let score = fact.importance;

      // 话题频次加成
      for (const tag of fact.tags) {
        const mentions = this.mentionTracker.get(tag) ?? 0;
        const newCount = mentions + 1;
        this.mentionTracker.set(tag, newCount);

        if (newCount >= this.config.minMentionsForHighImportance) {
          score += this.config.importanceBoostPerMention *
            (newCount - this.config.minMentionsForHighImportance + 1);
        }
      }

      // 内容长度因子：越长可能信息量越大（但过长的消息可能有噪音）
      const lengthFactor = Math.min(fact.content.length / 50, 1.5);
      score += lengthFactor * 0.1;

      // 类型加成：preference 和 decision 比普通 fact 更重要
      if (fact.type === 'preference') score += 0.15;
      if (fact.type === 'decision') score += 0.1;

      // 卡上限
      score = Math.min(score, 1.0);

      return { ...fact, importance: score };
    });
  }

  /**
   * 分类消息的类型
   */
  private classifyMessage(content: string): 'fact' | 'preference' | 'decision' {
    const lower = content.toLowerCase();

    // 偏好模式
    const preferencePatterns = [
      '我喜欢', '我讨厌', '我平时', '我习惯', '我更',
      'i like', 'i prefer', 'i usually', 'i tend to',
    ];

    // 决策模式
    const decisionPatterns = [
      '我决定', '我选择', '我用', '我打算',
      'i decided', 'i choose', 'i will use',
    ];

    for (const pat of preferencePatterns) {
      if (lower.includes(pat)) return 'preference';
    }

    for (const pat of decisionPatterns) {
      if (lower.includes(pat)) return 'decision';
    }

    return 'fact';
  }

  /**
   * 从文本中提取关键词作为标签
   *
   * 提取规则：
   * - 英文：长度 > 3 的名词性单词
   * - 中文：常见话题词（可通过词库扩展）
   */
  private extractTags(content: string): string[] {
    const tags = new Set<string>();

    // 英文单词（>3 字符）
    const words = content.toLowerCase().match(/[a-z]{4,}/g) ?? [];
    // 过滤掉常见停用词
    const stopWords = new Set([
      'this', 'that', 'with', 'from', 'have', 'been', 'what',
      'when', 'where', 'which', 'their', 'there',
    ]);
    for (const w of words) {
      if (!stopWords.has(w)) tags.add(w);
    }

    return Array.from(tags);
  }
}
