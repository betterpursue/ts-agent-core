import { describe, it, expect } from 'vitest';
import {
  ImportanceForgetting,
  AgeBasedForgetting,
  CapacityForgetting,
  CompositeForgetting,
  ForgetExecutor,
} from '../forgetting.js';
import type { LongTermMemoryItem } from '../forgetting.js';

function makeItem(
  id: string,
  opts: Partial<{ importance: number; timestamp: number; content: string }> = {},
): LongTermMemoryItem {
  return {
    id,
    content: opts.content ?? `记忆 ${id}`,
    metadata: {
      timestamp: opts.timestamp ?? Date.now(),
      type: 'fact',
      importance: opts.importance,
    },
  };
}

describe('ImportanceForgetting', () => {
  it('低重要性条目获得高分', async () => {
    const strategy = new ImportanceForgetting(0.5);
    const items = [makeItem('a', { importance: 0.2 }), makeItem('b', { importance: 0.8 })];
    const scores = await strategy.score(items);
    expect(scores[0].score).toBeGreaterThan(scores[1].score);
  });

  it('未设置重要性时按默认值 0.5 处理', async () => {
    const strategy = new ImportanceForgetting(0.5);
    const items = [makeItem('a')];
    const scores = await strategy.score(items);
    expect(scores[0].score).toBeCloseTo(0.5, 1);
  });
});

describe('AgeBasedForgetting', () => {
  it('超龄条目 score 为 1', async () => {
    const now = Date.now();
    const strategy = new AgeBasedForgetting(7 * 24 * 60 * 60 * 1000, 100);
    const items = [makeItem('a', { timestamp: now - 200 })];
    const scores = await strategy.score(items, { now, capacity: 0, currentSize: 1 });
    expect(scores[0].score).toBeCloseTo(1.0, 1);
  });

  it('半衰期处 score 应为 0.5', async () => {
    const now = Date.now();
    const halfLife = 7 * 24 * 60 * 60 * 1000;
    const strategy = new AgeBasedForgetting(halfLife, halfLife * 10);
    const items = [makeItem('a', { timestamp: now - halfLife })];
    const scores = await strategy.score(items, { now, capacity: 0, currentSize: 1 });
    expect(scores[0].score).toBeCloseTo(0.5, 1);
  });
});

describe('CapacityForgetting', () => {
  it('容量内条目 score 为 0', async () => {
    const strategy = new CapacityForgetting(5);
    const items = [
      makeItem('a', { importance: 0.9 }),
      makeItem('b', { importance: 0.8 }),
      makeItem('c', { importance: 0.7 }),
    ];
    const scores = await strategy.score(items);
    scores.forEach(s => expect(s.score).toBe(0));
  });

  it('溢出时最弱的条目 score 为 1', async () => {
    const strategy = new CapacityForgetting(2);
    const now = Date.now();
    const items = [
      makeItem('a', { importance: 0.9, timestamp: now }),
      makeItem('b', { importance: 0.8, timestamp: now }),
      makeItem('c', { importance: 0.7, timestamp: now }),
      makeItem('d', { importance: 0.6, timestamp: now }),
    ];
    const scores = await strategy.score(items);
    // 最弱的 d 应该 score = 1
    const dScore = scores.find(s => s.itemId === 'd')!;
    expect(dScore.score).toBeCloseTo(1.0, 1);
  });
});

describe('CompositeForgetting', () => {
  it('权重自动归一化', async () => {
    const strategy = new CompositeForgetting([
      { strategy: new ImportanceForgetting(0.5), weight: 1 },
      { strategy: new AgeBasedForgetting(7 * 24 * 60 * 60 * 1000, 9999999999999), weight: 1 },
    ]);
    const items = [makeItem('a', { importance: 0.5 })];
    const scores = await strategy.score(items, { now: Date.now(), capacity: 0, currentSize: 1 });
    // 归一化后各 0.5 权重，age score 接近 0，importance score = 0.5
    expect(scores[0].score).toBeCloseTo(0.25, 1);
  });
});

describe('ForgetExecutor', () => {
  it('删除超过阈值的条目', async () => {
    const memory = {
      getAllItems: () => [
        makeItem('keep', { importance: 0.9 }),
        makeItem('delete', { importance: 0.1 }),
      ],
      delete: async (id: string) => true,
      size: 2,
    };

    const executor = new ForgetExecutor({
      strategy: new ImportanceForgetting(0.5),
      scoreThreshold: 0.5,
    });

    const result = await executor.execute(memory as any);
    expect(result.deletedCount).toBe(1);
    expect(result.keptCount).toBe(1);
    expect(result.details.find(d => d.itemId === 'delete')!.deleted).toBe(true);
    expect(result.details.find(d => d.itemId === 'keep')!.deleted).toBe(false);
  });

  it('dry-run 不删除', async () => {
    const memory = {
      getAllItems: () => [makeItem('a', { importance: 0.1 })],
      delete: async (_id: string) => { throw new Error('should not call delete'); },
      size: 1,
    };

    const executor = new ForgetExecutor({
      strategy: new ImportanceForgetting(0.5),
      scoreThreshold: 0.5,
      dryRun: true,
    });

    const result = await executor.execute(memory as any);
    expect(result.deletedCount).toBe(0);
    expect(result.keptCount).toBe(1);
  });
});
