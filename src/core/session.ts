/**
 * 会话管理 — Agent 状态的持久化和恢复
 *
 * 设计原则：
 * - Session 隔离不同对话，每个 Session 有自己的记忆和上下文
 * - 支持序列化/反序列化（后续系列可扩展为 Redis/MySQL 持久化）
 * - Checkpoint 机制可以在任意时刻保存恢复
 */

import { randomUUID } from 'node:crypto';
import type { Message } from './message.js';

/** 会话状态快照 */
export interface SessionCheckpoint {
  id: string;
  timestamp: number;
  messages: Message[];
  metadata: Record<string, unknown>;
}

/** 会话 */
export interface Session {
  readonly id: string;
  readonly createdAt: number;

  /** 添加消息 */
  addMessage(message: Message): void;
  /** 获取所有消息 */
  getMessages(): Message[];
  /** 获取最近 N 条消息 */
  getRecentMessages(n: number): Message[];

  /** 保存一个检查点 */
  checkpoint(): SessionCheckpoint;
  /** 恢复到指定检查点 */
  restore(checkpoint: SessionCheckpoint): void;

  /** 序列化整个会话 */
  serialize(): string;
}

/** 简单的内存会话实现 */
export class InMemorySession implements Session {
  readonly id: string;
  readonly createdAt: number;
  private messages: Message[] = [];

  constructor(id?: string) {
    this.id = id ?? randomUUID();
    this.createdAt = Date.now();
  }

  addMessage(message: Message): void {
    this.messages.push(message);
  }

  getMessages(): Message[] {
    return [...this.messages];
  }

  getRecentMessages(n: number): Message[] {
    return this.messages.slice(-n);
  }

  checkpoint(): SessionCheckpoint {
    return {
      id: randomUUID(),
      timestamp: Date.now(),
      messages: [...this.messages],
      metadata: {},
    };
  }

  restore(checkpoint: SessionCheckpoint): void {
    this.messages = [...checkpoint.messages];
  }

  serialize(): string {
    return JSON.stringify({
      id: this.id,
      createdAt: this.createdAt,
      messages: this.messages,
    });
  }

  static deserialize(data: string): InMemorySession {
    const parsed = JSON.parse(data);
    const session = new InMemorySession(parsed.id);
    (session as unknown as { messages: Message[] }).messages = parsed.messages;
    return session;
  }
}
