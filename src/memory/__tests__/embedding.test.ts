/**
 * Embedding 模块测试
 *
 * 覆盖：
 * - cosineSimilarity 余弦相似度
 * - l2Normalize L2 归一化
 * - NoopEmbeddingProvider
 * - VectorIndex 向量索引
 * - InMemoryLongTermMemory 的 semantic / hybrid 检索模式
 */

import { describe, it, expect } from 'vitest';
import {
  cosineSimilarity,
  l2Normalize,
  NoopEmbeddingProvider,
} from '../embedding-provider.js';
import { VectorIndex } from '../vector-index.js';
import { InMemoryLongTermMemory } from '../long-term-memory.js';

// ─── cosineSimilarity ───────────────────────────────────────────

describe('cosineSimilarity', () => {
  it('returns 1 for identical vectors', () => {
    const a = [1, 0, 0];
    const b = [1, 0, 0];
    expect(cosineSimilarity(a, b)).toBeCloseTo(1, 5);
  });

  it('returns 0 for orthogonal vectors', () => {
    const a = [1, 0, 0];
    const b = [0, 1, 0];
    expect(cosineSimilarity(a, b)).toBeCloseTo(0, 5);
  });

  it('returns -1 for opposite vectors', () => {
    const a = [1, 0, 0];
    const b = [-1, 0, 0];
    expect(cosineSimilarity(a, b)).toBeCloseTo(-1, 5);
  });

  it('handles partial similarity', () => {
    const a = [1, 2, 3];
    const b = [2, 4, 6]; // 2x a → same direction
    expect(cosineSimilarity(a, b)).toBeCloseTo(1, 5);
  });

  it('handles zero vectors', () => {
    const a = [0, 0, 0];
    const b = [1, 0, 0];
    expect(cosineSimilarity(a, b)).toBe(0);
  });

  it('throws on dimension mismatch', () => {
    expect(() => cosineSimilarity([1, 2], [1, 2, 3])).toThrow(
      'Dimension mismatch'
    );
  });
});

// ─── l2Normalize ────────────────────────────────────────────────

describe('l2Normalize', () => {
  it('returns unit vector for non-zero input', () => {
    const v = [3, 4];
    const normalized = l2Normalize(v);
    expect(normalized[0]).toBeCloseTo(0.6, 5);
    expect(normalized[1]).toBeCloseTo(0.8, 5);

    // 验证模长为 1
    const norm = Math.sqrt(
      normalized[0] * normalized[0] + normalized[1] * normalized[1]
    );
    expect(norm).toBeCloseTo(1, 5);
  });

  it('preserves zero vector', () => {
    const v = [0, 0, 0];
    const normalized = l2Normalize(v);
    expect(normalized).toEqual([0, 0, 0]);
  });
});

// ─── NoopEmbeddingProvider ──────────────────────────────────────

describe('NoopEmbeddingProvider', () => {
  it('returns empty vectors', async () => {
    const provider = new NoopEmbeddingProvider();
    expect(await provider.embed('test')).toEqual([]);
    expect(await provider.embedMany(['a', 'b'])).toEqual([[], []]);
  });
});

// ─── VectorIndex ────────────────────────────────────────────────

describe('VectorIndex', () => {
  it('stores and retrieves vectors', () => {
    const idx = new VectorIndex(3);
    idx.add('a', [1, 0, 0]);

    const entry = idx.get('a');
    expect(entry).toBeDefined();
    expect(entry!.id).toBe('a');
  });

  it('returns nearest neighbors by cosine similarity', () => {
    const idx = new VectorIndex(2);
    idx.add('a', [1, 0]); // points right
    idx.add('b', [0, 1]); // points up
    idx.add('c', [0.9, 0.1]); // slightly up-right

    // query pointing right
    const results = idx.search([1, 0], 3);
    expect(results.length).toBe(3);
    expect(results[0].id).toBe('a'); // most similar
    expect(results[1].id).toBe('c'); // second
    expect(results[2].id).toBe('b'); // least similar
  });

  it('respects k parameter', () => {
    const idx = new VectorIndex(2);
    idx.add('a', [1, 0]);
    idx.add('b', [0, 1]);
    idx.add('c', [0.7, 0.7]);

    const results = idx.search([1, 0], 2);
    expect(results.length).toBe(2);
  });

  it('deletes entries', () => {
    const idx = new VectorIndex(2);
    idx.add('a', [1, 0]);
    expect(idx.size).toBe(1);

    idx.delete('a');
    expect(idx.size).toBe(0);
  });

  it('returns empty for empty index', () => {
    const idx = new VectorIndex(2);
    expect(idx.search([1, 0], 5)).toEqual([]);
  });

  it('returns empty for k <= 0', () => {
    const idx = new VectorIndex(2);
    idx.add('a', [1, 0]);
    expect(idx.search([1, 0], 0)).toEqual([]);
    expect(idx.search([1, 0], -1)).toEqual([]);
  });

  it('addMany adds multiple entries', () => {
    const idx = new VectorIndex(2);
    idx.addMany([
      { id: 'a', vector: [1, 0] },
      { id: 'b', vector: [0, 1] },
    ]);
    expect(idx.size).toBe(2);
  });

  it('clear empties the index', () => {
    const idx = new VectorIndex(2);
    idx.add('a', [1, 0]);
    idx.clear();
    expect(idx.size).toBe(0);
  });
});

// ─── InMemoryLongTermMemory — 语义检索模式 ─────────────────────

/**
 * Mock Embedding Provider 用于测试
 *
 * 用简单的规则模拟 embedding：
 * - "喜欢辣" → [1, 0, 1]（辣味方向）
 * - "喜欢编程" → [0, 1, 1]（技术方向）
 * - "Python" → [0, 1, 0.5]（偏技术）
 * - "天气" → [0.5, 0, 0]（无关方向）
 * - 其他 → [0, 0, 0]
 */
class MockEmbeddingProvider {
  readonly name = 'mock';
  private mapping = new Map([
    ['喜欢辣', [1, 0, 1]],
    ['喜欢辣的食物', [1, 0, 1]],
    ['用户喜欢吃辣的食物', [1, 0, 1]],
    ['喜欢编程和技术', [0, 1, 1]],
    ['用户喜欢编程和技术', [0, 1, 1]],
    ['Python 做后端', [0, 1, 0.5]],
    ['用户用 Python 做后端', [0, 1, 0.5]],
    ['今天天气不错', [0.5, 0, 0]],
  ]);

  async embed(text: string): Promise<number[]> {
    return this.mapping.get(text) ?? [0, 0, 0];
  }

  async embedMany(texts: string[]): Promise<number[][]> {
    return Promise.all(texts.map((t) => this.embed(t)));
  }
}

describe('InMemoryLongTermMemory — semantic mode', () => {
  it('stores embeddings automatically when provider is configured', async () => {
    const mem = new InMemoryLongTermMemory({
      embeddingProvider: new MockEmbeddingProvider(),
      searchMode: 'semantic',
      embeddingDimension: 3,
    });

    const id = await mem.store({
      content: '喜欢吃辣的食物',
      metadata: { timestamp: 1000, type: 'user_preference', importance: 0.8 },
    });

    const item = await mem.get(id);
    expect(item).not.toBeNull();
    expect(item!.embedding).toBeDefined();
    expect(item!.embedding!.length).toBe(3);
  });

  it('finds semantically similar items (synonyms)', async () => {
    const mem = new InMemoryLongTermMemory({
      embeddingProvider: new MockEmbeddingProvider(),
      searchMode: 'semantic',
      embeddingDimension: 3,
    });

    await mem.store({
      content: '用户喜欢吃辣的食物',
      metadata: { timestamp: 1000, type: 'user_preference', tags: ['food'], importance: 0.8 },
    });
    await mem.store({
      content: '用户喜欢编程和技术',
      metadata: { timestamp: 2000, type: 'fact', tags: ['tech'], importance: 0.6 },
    });
    await mem.store({
      content: '今天天气不错',
      metadata: { timestamp: 3000, type: 'fact', importance: 0.1 },
    });

    // 搜索"喜欢辣"——语义上应匹配第一条
    const results = await mem.search({
      query: '喜欢辣',
      maxResults: 5,
      minRelevance: 0.3,
    });

    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results[0].content).toContain('喜欢吃辣');
  });
});

// ─── InMemoryLongTermMemory — 混合检索模式 ─────────────────────

describe('InMemoryLongTermMemory — hybrid mode', () => {
  it('combines keyword and semantic search results', async () => {
    const mem = new InMemoryLongTermMemory({
      embeddingProvider: new MockEmbeddingProvider(),
      searchMode: 'hybrid',
      embeddingDimension: 3,
    });

    await mem.store({
      content: '用户喜欢吃辣的食物',
      metadata: { timestamp: 1000, type: 'user_preference', tags: ['food'], importance: 0.8 },
    });
    await mem.store({
      content: '用户喜欢编程和技术',
      metadata: { timestamp: 2000, type: 'fact', tags: ['tech'], importance: 0.6 },
    });
    await mem.store({
      content: '今天天气不错',
      metadata: { timestamp: 3000, type: 'fact', importance: 0.1 },
    });

    // hybrid 搜索，keyword 和 semantic 各贡献排序
    const results = await mem.search({
      query: '用户喜欢吃辣的食物',
      maxResults: 5,
    });

    expect(results.length).toBeGreaterThanOrEqual(1);
    // 最相关的结果应包含"辣"
    expect(results.some((r) => r.content.includes('辣'))).toBe(true);
  });
});

// ─── 回归：关键词检索保持不变 ──────────────────────────────────

describe('InMemoryLongTermMemory — keyword mode (regression)', () => {
  it('still works without embedding provider', async () => {
    const mem = new InMemoryLongTermMemory();

    await mem.store({
      content: '用户喜欢吃辣的食物',
      metadata: { timestamp: 1000, type: 'user_preference', importance: 0.8 },
    });

    const results = await mem.search({ query: '喜欢', maxResults: 10 });
    expect(results.length).toBe(1);
  });

  it('default constructor uses keyword mode', () => {
    const mem = new InMemoryLongTermMemory();
    // 验证没有 embedding provider 也不报错
    expect(mem).toBeDefined();
  });

  it('does not generate embeddings without provider', async () => {
    const mem = new InMemoryLongTermMemory();
    const id = await mem.store({
      content: 'test',
      metadata: { timestamp: 1000, type: 'fact' },
    });
    const item = await mem.get(id);
    expect(item!.embedding).toBeUndefined();
  });
});
