/**
 * 记忆模块 — Agent 的长短期记忆抽象
 *
 * 设计原则：
 * - 记忆分为短期（滑动窗口）和长期（检索增强）两层
 * - 每层都是接口，可以有不同实现
 * - 后续系列可以实现 RedisBackedLongTermMemory、SQLiteMemory 等
 */

import type { Message } from './message.js';

/** 短期记忆 —— 最近的对话上下文 */
export interface ShortTermMemory {
  /** 添加消息到短期记忆 */
  add(message: Message): void;
  /** 获取当前短期记忆中的所有消息 */
  getMessages(): Message[];
  /** 清理（重置） */
  clear(): void;
  /** 当前 token 估算值 */
  estimatedTokens(): number;
}

/** 长期记忆存储项 */
export interface LongTermMemoryItem {
  id: string;
  content: string;
  metadata: {
    timestamp: number;
    type: 'summary' | 'fact' | 'insight' | 'user_preference';
    tags?: string[];
    importance?: number; // 0-1，用于遗忘策略
  };
  embedding?: number[];
}

/** 长期记忆检索结果 */
export interface LongTermMemoryQuery {
  query: string;
  maxResults?: number;
  minRelevance?: number;
  tagFilter?: string[];
}

/** 长期记忆 —— 持久化的知识存储 */
export interface LongTermMemory {
  /** 存储一条信息 */
  store(item: Omit<LongTermMemoryItem, 'id'>): Promise<string>;
  /** 检索相关信息 */
  search(query: LongTermMemoryQuery): Promise<LongTermMemoryItem[]>;
  /** 按 ID 获取 */
  get(id: string): Promise<LongTermMemoryItem | null>;
  /** 删除 */
  delete(id: string): Promise<boolean>;
  /** 清理低重要性记忆（遗忘机制入口） */
  forget(threshold: number): Promise<number>;
}

/** 记忆注入时机 */
export type InjectionTiming = 'before_first_call' | 'before_every_call';

/** 记忆注入配置 */
export interface MemoryInjectionConfig {
  /** 每次注入的最大记忆条数（默认 5） */
  maxMemories?: number;
  /** 最小相关度阈值（默认 0.15） */
  minRelevance?: number;
  /** 注入时机（默认 before_every_call） */
  timing?: InjectionTiming;
  /** 构建检索 query 时使用的最近消息窗口大小（默认 3） */
  queryWindowSize?: number;
}

/**
 * 记忆注入器 —— 将长期记忆注入到 LLM 上下文中
 *
 * 职责：
 * 1. 根据当前对话构造检索 query
 * 2. 从长期记忆中检索相关条目
 * 3. 格式化为 LLM 可读的上下文文本
 */
export interface MemoryInjector {
  /**
   * 构建记忆上下文文本
   *
   * @param messages 当前完整对话消息列表
   * @param longTerm 长期记忆存储
   * @param config 注入配置
   * @returns 格式化的记忆上下文文本，空字符串表示没有相关记忆
   */
  buildContext(
    messages: import('./message.js').Message[],
    longTerm: LongTermMemory,
    config?: MemoryInjectionConfig
  ): Promise<string>;
}

/** 完整的 Agent 记忆系统 */
export interface MemorySystem {
  shortTerm: ShortTermMemory;
  longTerm: LongTermMemory;

  /** 将短期记忆中的关键信息提炼到长期记忆 */
  consolidate(): Promise<void>;
}

/** 简单的滑动窗口短期记忆 */
export class SlidingWindowMemory implements ShortTermMemory {
  private messages: Message[] = [];
  private maxTokens: number;

  constructor(maxTokens = 4096) {
    this.maxTokens = maxTokens;
  }

  add(message: Message): void {
    this.messages.push(message);
    this.evict();
  }

  getMessages(): Message[] {
    return [...this.messages];
  }

  clear(): void {
    this.messages = [];
  }

  estimatedTokens(): number {
    // 粗略估算：每 4 个字符 ≈ 1 token
    return this.messages.reduce((sum, m) => sum + Math.ceil(m.content.length / 4), 0);
  }

  private evict(): void {
    while (this.estimatedTokens() > this.maxTokens && this.messages.length > 1) {
      // 保留系统提示，移除最早的对话消息
      const firstUserOrAssistant = this.messages.findIndex(
        (m) => m.role === 'user' || m.role === 'assistant'
      );
      if (firstUserOrAssistant === -1) break;
      this.messages.splice(firstUserOrAssistant, 2); // 移除一对 user/assistant
    }
  }
}
