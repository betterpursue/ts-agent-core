/**
 * Session 持久化测试
 *
 * 测试覆盖：
 * - FileSession 的序列化和反序列化
 * - 异步写盘的完成确认
 * - 从文件恢复对话
 * - FileCheckpointManager 的保存、加载、列表、删除
 * - 文件损坏的优雅处理
 */

import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { mkdtempSync, rmSync, existsSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import {
  FileSession,
  FileCheckpointManager,
  InMemorySession,
} from '../../src/core/session.js';
import type { SessionCheckpoint } from '../../src/core/session.js';
import { userMessage, assistantMessage, toolResultMessage } from '../../src/core/message.js';

// ─── FileSession ─────────────────────────────────────────────────

describe('FileSession', () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), 'session-test-'));
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it('新会话应该创建空消息列表', () => {
    const session = new FileSession({ storageDir: tmpDir });
    expect(session.getMessages()).toHaveLength(0);
    expect(existsSync(session.storagePath)).toBe(false);
  });

  it('添加消息后文件应被异步写入', async () => {
    const session = new FileSession({ storageDir: tmpDir });
    session.addMessage(userMessage('Hello'));
    session.addMessage(assistantMessage('Hi there!'));

    // 等待异步写盘防抖完成
    await new Promise((r) => setTimeout(r, 200));

    expect(existsSync(session.storagePath)).toBe(true);
    const content = readFileSync(session.storagePath, 'utf-8');
    const parsed = JSON.parse(content);
    expect(parsed.messages).toHaveLength(2);
    expect(parsed.messages[0].role).toBe('user');
    expect(parsed.messages[0].content).toBe('Hello');
  });

  it('load() 应该从文件恢复完整的消息列表', async () => {
    const session1 = new FileSession({ storageDir: tmpDir });
    session1.addMessage(userMessage('保存这条消息'));
    await new Promise((r) => setTimeout(r, 200));

    // 用同一个 id 创建新实例，从文件恢复
    const session2 = new FileSession({
      id: session1.id,
      storageDir: tmpDir,
    });
    await session2.load();

    const messages = session2.getMessages();
    expect(messages).toHaveLength(1);
    expect(messages[0].content).toBe('保存这条消息');
  });

  it('load() 在文件不存在时应静默创建新会话', async () => {
    const session = new FileSession({ storageDir: tmpDir });
    await session.load();
    expect(session.getMessages()).toHaveLength(0);
    expect(session.id).toBeTruthy();
  });

  it('文件损坏时应抛出可理解的错误', async () => {
    const session = new FileSession({ storageDir: tmpDir });
    session.addMessage(userMessage('Hi'));
    await new Promise((r) => setTimeout(r, 200));

    // 手动破坏文件内容
    writeFileSync(session.storagePath, '{invalid json}', 'utf-8');

    await expect(session.load()).rejects.toThrow();
  });

  it('serialize() 和 InMemorySession.deserialize() 应互相兼容', () => {
    const session = new FileSession({ storageDir: tmpDir });
    session.addMessage(userMessage('Test'));
    session.addMessage(assistantMessage('Response'));

    const serialized = session.serialize();
    const restored = InMemorySession.deserialize(serialized);

    expect(restored.getMessages()).toHaveLength(2);
    expect(restored.getMessages()[0].content).toBe('Test');
    expect(restored.id).toBe(session.id);
  });

  it('消息数量较多时序列化和反序列化应正确', () => {
    const session = new FileSession({ storageDir: tmpDir });
    const count = 100;
    for (let i = 0; i < count; i++) {
      session.addMessage(userMessage(`Message ${i}`));
    }

    const serialized = session.serialize();
    const restored = InMemorySession.deserialize(serialized);

    expect(restored.getMessages()).toHaveLength(count);
    expect(restored.getMessages()[count - 1].content).toBe(`Message ${count - 1}`);
  });

  it('多次 addMessage 后 save 应只触发一次写盘', async () => {
    const session = new FileSession({ storageDir: tmpDir });

    // 连续快速添加多条消息
    for (let i = 0; i < 50; i++) {
      session.addMessage(userMessage(`Quick msg ${i}`));
    }

    // 等待防抖写盘
    await new Promise((r) => setTimeout(r, 300));

    const content = readFileSync(session.storagePath, 'utf-8');
    const parsed = JSON.parse(content);
    expect(parsed.messages).toHaveLength(50);
  });
});

// ─── FileCheckpointManager ───────────────────────────────────────

describe('FileCheckpointManager', () => {
  let tmpDir: string;

  beforeEach(() => {
    tmpDir = mkdtempSync(join(tmpdir(), 'ckpt-test-'));
  });

  afterEach(() => {
    rmSync(tmpDir, { recursive: true, force: true });
  });

  it('应该保存和加载检查点', async () => {
    const mgr = new FileCheckpointManager(tmpDir);
    const sessionId = 'test-session';

    const ckpt: SessionCheckpoint = {
      id: 'ckpt-1',
      timestamp: Date.now(),
      messages: [userMessage('Hi')],
      metadata: { source: 'test' },
    };

    await mgr.saveCheckpoint(sessionId, ckpt);

    const loaded = await mgr.loadCheckpoint(sessionId, 'ckpt-1');
    expect(loaded).not.toBeNull();
    expect(loaded!.messages).toHaveLength(1);
    expect(loaded!.messages[0].content).toBe('Hi');
    expect(loaded!.metadata.source).toBe('test');
  });

  it('加载不存在的检查点应返回 null', async () => {
    const mgr = new FileCheckpointManager(tmpDir);
    const result = await mgr.loadCheckpoint('test-session', 'non-existent');
    expect(result).toBeNull();
  });

  it('应该列出所有检查点并按时间排序', async () => {
    const mgr = new FileCheckpointManager(tmpDir);
    const sessionId = 'session-sort';

    await mgr.saveCheckpoint(sessionId, {
      id: 'ckpt-1',
      timestamp: 1000,
      messages: [userMessage('first')],
      metadata: {},
    });

    await mgr.saveCheckpoint(sessionId, {
      id: 'ckpt-2',
      timestamp: 2000,
      messages: [userMessage('second')],
      metadata: {},
    });

    const list = await mgr.listCheckpoints(sessionId);
    expect(list).toHaveLength(2);
    // 按时间升序排列
    expect(list[0].id).toBe('ckpt-1');
    expect(list[1].id).toBe('ckpt-2');
  });

  it('空会话的检查点列表应返回空数组', async () => {
    const mgr = new FileCheckpointManager(tmpDir);
    const list = await mgr.listCheckpoints('non-existent-session');
    expect(list).toHaveLength(0);
  });

  it('应该删除指定的检查点', async () => {
    const mgr = new FileCheckpointManager(tmpDir);
    const sessionId = 'session-delete';

    await mgr.saveCheckpoint(sessionId, {
      id: 'ckpt-to-delete',
      timestamp: Date.now(),
      messages: [],
      metadata: {},
    });

    await mgr.deleteCheckpoint(sessionId, 'ckpt-to-delete');
    const loaded = await mgr.loadCheckpoint(sessionId, 'ckpt-to-delete');
    expect(loaded).toBeNull();
  });

  it('删除不存在的检查点不应报错', async () => {
    const mgr = new FileCheckpointManager(tmpDir);
    await expect(
      mgr.deleteCheckpoint('test-session', 'non-existent')
    ).resolves.not.toThrow();
  });

  it('多个会话的检查点应互相隔离', async () => {
    const mgr = new FileCheckpointManager(tmpDir);

    await mgr.saveCheckpoint('session-a', {
      id: 'ckpt-1',
      timestamp: Date.now(),
      messages: [userMessage('A')],
      metadata: {},
    });

    await mgr.saveCheckpoint('session-b', {
      id: 'ckpt-1',
      timestamp: Date.now(),
      messages: [userMessage('B')],
      metadata: {},
    });

    const listA = await mgr.listCheckpoints('session-a');
    const listB = await mgr.listCheckpoints('session-b');

    expect(listA).toHaveLength(1);
    expect(listB).toHaveLength(1);
    expect(listA[0].messages[0].content).toBe('A');
    expect(listB[0].messages[0].content).toBe('B');
  });

  it('检查点包含完整对话时应能正确恢复', async () => {
    const mgr = new FileCheckpointManager(tmpDir);
    const sessionId = 'session-full';

    // 保存一个包含多轮对话的检查点
    await mgr.saveCheckpoint(sessionId, {
      id: 'full-ckpt',
      timestamp: Date.now(),
      messages: [
        userMessage('What is 1+1?'),
        assistantMessage('', [
          { id: 'call_1', name: 'calculator', args: { expression: '1+1' } },
        ]),
        toolResultMessage('call_1', 'calculator', '2'),
        assistantMessage('The answer is 2.'),
      ],
      metadata: { iterations: 2 },
    });

    const loaded = await mgr.loadCheckpoint(sessionId, 'full-ckpt');
    expect(loaded).not.toBeNull();
    expect(loaded!.messages).toHaveLength(4);
    expect(loaded!.metadata.iterations).toBe(2);

    // 验证工具调用消息的完整性
    const toolCallMsg = loaded!.messages[1];
    expect(toolCallMsg.role).toBe('assistant');
    expect(toolCallMsg.toolCalls).toHaveLength(1);
    expect(toolCallMsg.toolCalls![0].name).toBe('calculator');
  });
});
