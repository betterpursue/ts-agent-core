/**
 * DefaultMemoryInjector — 将长期记忆注入到 LLM 上下文的默认实现
 *
 * 设计原则：
 * - 不修改消息列表，只生成一段格式化文本
 * - 由 Agent 执行循环决定插入位置和时机
 * - 零外部依赖，纯 TypeScript
 * - 检索 query 从最近的 N 条消息中推断
 *
 * 核心逻辑：
 * 1. 从最近的对话消息中构造检索 query
 * 2. 查询长期记忆，获取最相关的条目
 * 3. 格式化为 LLM 可读的记忆上下文文本
 * 4. 返回空字符串表示没有相关记忆
 */

import type { Message } from '../core/message.js';
import type {
  LongTermMemory,
  MemoryInjector,
  MemoryInjectionConfig,
} from '../core/memory.js';

/** 默认配置 */
const DEFAULT_CONFIG: Required<MemoryInjectionConfig> = {
  maxMemories: 5,
  minRelevance: 0.15,
  timing: 'before_every_call',
  queryWindowSize: 3,
};

/**
 * 默认记忆注入器
 *
 * 使用方式：
 * ```typescript
 * const injector = new DefaultMemoryInjector();
 *
 * const context = await injector.buildContext(messages, longTerm, {
 *   maxMemories: 5,
 *   minRelevance: 0.15,
 * });
 * ```
 */
export class DefaultMemoryInjector implements MemoryInjector {
  async buildContext(
    messages: Message[],
    longTerm: LongTermMemory,
    config?: MemoryInjectionConfig
  ): Promise<string> {
    const cfg = { ...DEFAULT_CONFIG, ...config };
    const queryText = this.buildQuery(messages, cfg.queryWindowSize);

    if (!queryText) {
      return ''; // 没有足够的对话来构建 query
    }

    const results = await longTerm.search({
      query: queryText,
      maxResults: cfg.maxMemories,
      minRelevance: cfg.minRelevance,
    });

    if (results.length === 0) {
      return '';
    }

    return this.formatMemoryContext(results);
  }

  /**
   * 从最近的 N 条非系统消息中构建检索 query
   *
   * 策略：
   * - 跳过 system 消息（系统提示不包含语义 query 信息）
   * - 取最近的 queryWindowSize 条 user + assistant 消息
   * - 拼接成一段连续的文本
   * - 优先提取用户消息的内容（用户的表述是更好的检索信号）
   *
   * 为什么用最近 N 条：
   * - 对话越长，全部拼接的 token 开销越大
   * - 最近的对话通常反映当前话题
   * - 3 条是一个经验值，足够捕获话题信号又不过度
   */
  private buildQuery(
    messages: Message[],
    windowSize: number
  ): string {
    // 过滤掉 system 和 tool 消息
    const relevantMessages = messages.filter(
      (m) => m.role === 'user' || m.role === 'assistant'
    );

    if (relevantMessages.length === 0) return '';

    // 取最近的 windowSize 条
    const window = relevantMessages.slice(-windowSize);

    // 按 User → Assistant 顺序拼接，用换行分隔
    return window
      .map((m) => m.content)
      .filter((c) => c.length > 0)
      .join('\n');
  }

  /**
   * 将检索到的记忆条目格式化为 LLM 友好的上下文文本
   *
   * 格式设计：
   * ```
   * [记忆上下文 - 以下是你需要知道的用户信息]
   * - 📝 事实: 用户住在北京 (重要度: 0.8)
   * - ❤️ 偏好: 用户喜欢吃辣 (重要度: 0.75)
   * - 💡 洞察: 用户决定用 TypeScript 重写后端 (重要度: 0.6)
   * ```
   *
   * 前缀 emoji 帮助 LLM 快速区分记忆类型。
   * 时间戳有助于 LLM 判断信息时效性。
   * 不加过多格式，减少 token 浪费。
   */
  private formatMemoryContext(
    items: Array<{
      content: string;
      metadata: {
        type: 'summary' | 'fact' | 'insight' | 'user_preference';
        timestamp: number;
        importance?: number;
      };
    }>
  ): string {
    const typeLabel: Record<string, string> = {
      fact: '📝 事实',
      user_preference: '❤️ 偏好',
      insight: '💡 洞察',
      summary: '📋 摘要',
    };

    const lines = items.map((item) => {
      const label = typeLabel[item.metadata.type] ?? '📌 信息';
      const time = this.formatTimestamp(item.metadata.timestamp);
      const importance = item.metadata.importance?.toFixed(2) ?? 'N/A';

      return `- ${label}: ${item.content} (${time}, 重要度: ${importance})`;
    });

    return [
      '[记忆上下文 —— 以下是你需要知道的用户信息]',
      ...lines,
      '---',
    ].join('\n');
  }

  /**
   * 格式化时间戳为可读形式
   *
   * 相对时间（最近 24 小时内显示"X 小时前"）
   * 绝对时间（超过 24 小时显示日期）
   */
  private formatTimestamp(ts: number): string {
    const diff = Date.now() - ts;
    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) {
      const date = new Date(ts);
      return `${date.getMonth() + 1}/${date.getDate()}`;
    }
    if (hours > 0) return `${hours} 小时前`;
    if (minutes > 0) return `${minutes} 分钟前`;
    return '刚刚';
  }
}

/**
 * 不注入任何记忆的注入器（空实现）
 *
 * 在测试或特定场景下使用，统一接口但不执行注入。
 * 遵循 Null Object Pattern。
 */
export class NoopMemoryInjector implements MemoryInjector {
  async buildContext(): Promise<string> {
    return '';
  }
}
