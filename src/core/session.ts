/**
 * 会话管理 — Agent 状态的持久化和恢复
 *
 * 设计原则：
 * - Session 隔离不同对话，每个 Session 有自己的记忆和上下文
 * - 支持序列化/反序列化（后续系列可扩展为 Redis/MySQL 持久化）
 * - Checkpoint 机制可以在任意时刻保存恢复
 * - PersistentSession 提供持久化语义，文件/Redis/MySQL 实现可互换
 */

import { randomUUID } from 'node:crypto';
import { readFile, writeFile, mkdir, readdir, unlink } from 'node:fs/promises';
import { dirname } from 'node:path';
import type { Message } from './message.js';

// ─── 基础会话 ───────────────────────────────────────────────────

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
  /** @internal 子类需要直接访问消息数组 */
  protected messages: Message[] = [];

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

// ─── 持久化会话 ─────────────────────────────────────────────────

/**
 * 持久的会话 —— 在 Session 基础上增加 save()/load()
 *
 * 后续系列（Redis、MySQL）只需要实现此接口即可接入 Agent。
 */
export interface PersistentSession extends Session {
  /** 保存到持久化存储 */
  save(): Promise<void>;
  /** 从持久化存储恢复 */
  load(): Promise<void>;
  /** 持久化的存储路径/标识 */
  readonly storagePath: string;
}

/** 检查点触发策略 */
export type CheckpointTrigger =
  | { type: 'interval'; messageCount: number }
  | { type: 'tool_call' }
  | { type: 'manual' };

/**
 * 基于文件系统的持久化会话
 *
 * 设计：
 * - 读/写全部走内存，零开销
 * - 每次 addMessage 触发异步防抖写盘（write-through）
 * - load() 从文件恢复，文件不存在时静默创建新会话
 * - 存储结构：{storageDir}/{sessionId}.session.json
 */
export class FileSession extends InMemorySession implements PersistentSession {
  readonly storagePath: string;
  private dirty = false;
  private savePromise: Promise<void> | null = null;

  constructor(options: {
    id?: string;
    storageDir: string;
  }) {
    super(options.id);
    this.storagePath = `${options.storageDir}/${this.id}.session.json`;
  }

  /**
   * 从文件恢复会话
   * 文件不存在时静默创建新会话
   */
  async load(): Promise<void> {
    try {
      const data = await readFile(this.storagePath, 'utf-8');
      const parsed = JSON.parse(data);

      if (!Array.isArray(parsed?.messages)) {
        throw new Error(
          `Corrupted session file: ${this.storagePath} — 'messages' must be an array`
        );
      }

      this.messages = parsed.messages;
      (this as unknown as { createdAt: number }).createdAt = parsed.createdAt ?? Date.now();
    } catch (err: unknown) {
      if ((err as NodeJS.ErrnoException)?.code === 'ENOENT') {
        return;
      }
      throw err;
    }
  }

  /**
   * 保存到文件
   * 使用防抖机制：连续快速调用只触发一次写盘
   */
  async save(): Promise<void> {
    this.dirty = true;
    if (this.savePromise) return;

    this.savePromise = this.flush().finally(() => {
      this.savePromise = null;
    });
    return this.savePromise;
  }

  private async flush(): Promise<void> {
    if (!this.dirty) return;
    this.dirty = false;

    const dir = dirname(this.storagePath);
    await mkdir(dir, { recursive: true });

    const data = JSON.stringify({
      id: this.id,
      createdAt: (this as unknown as { createdAt: number }).createdAt,
      messages: this.messages,
    }, null, 2);

    await writeFile(this.storagePath, data, 'utf-8');
  }

  /**
   * 带自动保存的消息添加
   * 每次添加消息后标记 dirty 并异步触发写盘
   */
  addMessage(message: Message): void {
    super.addMessage(message);
    this.save().catch((err) => {
      console.error(`[FileSession] Save failed for ${this.id}:`, err);
    });
  }
}

// ─── 检查点管理 ─────────────────────────────────────────────────

/**
 * 检查点管理器接口
 *
 * 不同后端只需实现此接口：
 * - FileCheckpointManager：文件系统
 * - RedisCheckpointManager（后续系列）：Redis LIST
 * - MySQLCheckpointManager（后续系列）：MySQL table
 */
export interface CheckpointManager {
  /** 保存一个检查点到持久化存储 */
  saveCheckpoint(sessionId: string, checkpoint: SessionCheckpoint): Promise<void>;
  /** 列出某个会话的所有检查点 */
  listCheckpoints(sessionId: string): Promise<SessionCheckpoint[]>;
  /** 加载指定检查点 */
  loadCheckpoint(sessionId: string, checkpointId: string): Promise<SessionCheckpoint | null>;
  /** 删除指定检查点 */
  deleteCheckpoint(sessionId: string, checkpointId: string): Promise<void>;
}

/**
 * 基于文件系统的检查点管理器
 *
 * 存储结构：
 *   {baseDir}/{sessionId}/checkpoints/{checkpointId}.ckpt.json
 */
export class FileCheckpointManager implements CheckpointManager {
  private baseDir: string;

  constructor(baseDir: string) {
    this.baseDir = baseDir;
  }

  async saveCheckpoint(
    sessionId: string,
    checkpoint: SessionCheckpoint
  ): Promise<void> {
    const dir = `${this.baseDir}/${sessionId}/checkpoints`;
    await mkdir(dir, { recursive: true });
    const path = `${dir}/${checkpoint.id}.ckpt.json`;
    await writeFile(path, JSON.stringify(checkpoint, null, 2), 'utf-8');
  }

  async listCheckpoints(sessionId: string): Promise<SessionCheckpoint[]> {
    const dir = `${this.baseDir}/${sessionId}/checkpoints`;
    try {
      const files = await readdir(dir);
      const checkpoints: SessionCheckpoint[] = [];
      for (const file of files) {
        if (file.endsWith('.ckpt.json')) {
          const data = await readFile(`${dir}/${file}`, 'utf-8');
          checkpoints.push(JSON.parse(data));
        }
      }
      checkpoints.sort((a, b) => a.timestamp - b.timestamp);
      return checkpoints;
    } catch {
      return [];
    }
  }

  async loadCheckpoint(
    sessionId: string,
    checkpointId: string
  ): Promise<SessionCheckpoint | null> {
    const path = `${this.baseDir}/${sessionId}/checkpoints/${checkpointId}.ckpt.json`;
    try {
      const data = await readFile(path, 'utf-8');
      return JSON.parse(data);
    } catch {
      return null;
    }
  }

  async deleteCheckpoint(
    sessionId: string,
    checkpointId: string
  ): Promise<void> {
    const path = `${this.baseDir}/${sessionId}/checkpoints/${checkpointId}.ckpt.json`;
    try {
      await unlink(path);
    } catch {
      // 文件不存在也视为成功删除
    }
  }
}

/**
 * 带保留策略的检查点管理器（装饰器）
 *
 * 自动清理最旧的 checkpoint，只保留最近 N 个。
 * 不需要修改核心实现，通过装饰器组合。
 */
export class RetentionCheckpointManager implements CheckpointManager {
  constructor(
    private inner: CheckpointManager,
    private maxCheckpoints: number
  ) {}

  async saveCheckpoint(sessionId: string, checkpoint: SessionCheckpoint): Promise<void> {
    await this.inner.saveCheckpoint(sessionId, checkpoint);

    const all = await this.inner.listCheckpoints(sessionId);
    if (all.length > this.maxCheckpoints) {
      const toDelete = all.slice(0, all.length - this.maxCheckpoints);
      for (const c of toDelete) {
        await this.inner.deleteCheckpoint(sessionId, c.id);
      }
    }
  }

  async listCheckpoints(sessionId: string): Promise<SessionCheckpoint[]> {
    return this.inner.listCheckpoints(sessionId);
  }

  async loadCheckpoint(sessionId: string, checkpointId: string): Promise<SessionCheckpoint | null> {
    return this.inner.loadCheckpoint(sessionId, checkpointId);
  }

  async deleteCheckpoint(sessionId: string, checkpointId: string): Promise<void> {
    return this.inner.deleteCheckpoint(sessionId, checkpointId);
  }
}
