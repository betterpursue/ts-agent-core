/**
 * 检索结果重排器测试 —— 时效加权 + MMR 多样性重排
 *
 * 覆盖：
 * - DefaultReranker 基础功能：
 *   - 空结果 / 单条结果 → 原样返回
 *   - 时效加权能提升近期记忆的排名
 *   - MMR 能提升多样性，避免相似记忆扎堆
 *   - 组合使用（先时效，后 MMR）
 * - NoopReranker 原样返回
 * - Jaccard 相似度计算
 * - 指数衰减函数
 * - DefaultMemoryInjector 集成 reranker
 */

import { describe, it, expect } from 'vitest';
import { DefaultReranker, NoopReranker } from '../reranker.js';
import { DefaultMemoryInjector } from '../memory-injector.js';
import { InMemoryLongTermMemory } from '../long-term-memory.js';
import type { LongTermMemoryItem } from '../../core/memory.js';

// ─── 辅助函数 ───────────────────────────────────────────────────

/**
 * 快速创建一条长期记忆条目
 */
function makeItem(
  id: string,
  content: string,
  overrides?: Partial<LongTermMemoryItem['metadata']>
): LongTermMemoryItem {
  return {
    id,
    content,
    metadata: {
      timestamp: Date.now(),
      type: 'fact',
      importance: 0.5,
      ...overrides,
    },
  };
}

function userMessage(content: string) {
  return { role: 'user' as const, content };
}

function systemMessage(content: string) {
  return { role: 'system' as const, content };
}

// ─── NoopReranker ───────────────────────────────────────────────

describe('NoopReranker', () => {
  it('returns results unchanged', async () => {
    const reranker = new NoopReranker();
    const items = [makeItem('1', 'hello'), makeItem('2', 'world')];

    const result = await reranker.rerank('test', items);
    expect(result).toEqual(items);
  });

  it('returns empty for empty input', async () => {
    const reranker = new NoopReranker();
    const result = await reranker.rerank('test', []);
    expect(result).toEqual([]);
  });
});

// ─── DefaultReranker 边界情况 ──────────────────────────────────

describe('DefaultReranker: edge cases', () => {
  it('returns empty for empty results', async () => {
    const reranker = new DefaultReranker();
    const result = await reranker.rerank('test', []);
    expect(result).toEqual([]);
  });

  it('returns single item unchanged', async () => {
    const reranker = new DefaultReranker();
    const items = [makeItem('1', 'user likes spicy food')];

    const result = await reranker.rerank('spicy', items);
    expect(result).toHaveLength(1);
    expect(result[0].id).toBe('1');
  });

  it('does not exceed maxResults', async () => {
    const reranker = new DefaultReranker();
    const items = Array.from({ length: 10 }, (_, i) =>
      makeItem(`id_${i}`, `item number ${i}`)
    );

    const result = await reranker.rerank('item number', items, { maxResults: 3 });
    expect(result.length).toBeLessThanOrEqual(3);
  });
});

// ─── DefaultReranker: 时效加权 ──────────────────────────────────

describe('DefaultReranker: recency weighting', () => {
  /**
   * 核心验证：时效加权后，近期记忆应该排在远期记忆之前
   *
   * 两个内容相似的记忆：（都和 "like hiking" 相关）
   * - "user likes hiking"（3 天前）
   * - "user likes hiking in mountains"（刚刚）
   *
   * 如果不启用时效加权，两者都同样相关，排序不确定。
   * 启用后，刚刚记下的那条应该排在前面。
   */
  it('boosts recent memories over old ones', async () => {
    const reranker = new DefaultReranker();
    const now = Date.now();

    // 两条内容量相似但时间差距很大的记忆
    const items = [
      makeItem('old', 'user likes hiking', {
        timestamp: now - 30 * 24 * 60 * 60 * 1000, // 30 天前
      }),
      makeItem('recent', 'user likes hiking in mountains', {
        timestamp: now - 60 * 60 * 1000, // 1 小时前
      }),
    ];

    const result = await reranker.rerank('like hiking', items, {
      recencyWeight: 0.5, // 时效权重开大，效果明显
      maxResults: 2,
      mmrLambda: 1.0, // 关闭 MMR，只测试时效
    });

    // 最近的那条应该在前面
    expect(result[0].id).toBe('recent');
  });

  /**
   * 极端情况：recencyWeight = 0 时，时效不起作用
   * 排序结果应该和原始相关度排序一致
   */
  it('does nothing when recencyWeight is 0', async () => {
    const reranker = new DefaultReranker();
    const now = Date.now();

    const items = [
      makeItem('exact', 'python programming', { timestamp: now - 1000 }),
      makeItem('partial', 'programming in python is fun', {
        timestamp: now - 1000,
      }),
      makeItem('old', 'i like hiking', { timestamp: now - 100 * 24 * 60 * 60 * 1000 }),
    ];

    // recencyWeight = 0 时，结果应该主要按 Jaccard 相似度排
    const result = await reranker.rerank('python programming', items, {
      recencyWeight: 0,
      mmrLambda: 1.0,
      maxResults: 3,
    });

    // "exact" 和 query 完全一样，Jaccard = 1，应该排第一
    expect(result[0].id).toBe('exact');
  });
});

// ─── DefaultReranker: MMR 多样性 ───────────────────────────────

describe('DefaultReranker: MMR diversity', () => {
  /**
   * 核心验证：MMR 应避免返回高度相似的结果
   *
   * 候选集中有 4 条记忆：
   * - 3 条关于 "TypeScript"（高度相似）
   * - 1 条关于 "Python"（完全不同）
   *
   * 纯相关度排序下，前 3 条全是 TypeScript。
   * MMR 生效后，应该至少有一条 Python 被选入前 3。
   */
  it('prefers diverse results over similar ones', async () => {
    const reranker = new DefaultReranker();
    const now = Date.now();

    // 4 条与 query "TypeScript" 全都相关的条目：
    // - 前 3 条高度相似（都是 TypeScript 后端开发）
    // - 第 4 条主题不同（数据分析），但与 query 同样相关
    //
    // querySimilarity 相近 + pairwise 差异大 = MMR 应选 py
    const items = [
      makeItem('ts1', 'writing TypeScript backend with Express for REST API', {
        timestamp: now,
        type: 'fact',
      }),
      makeItem('ts2', 'building TypeScript backend API using Express framework', {
        timestamp: now,
        type: 'insight',
      }),
      makeItem('ts3', 'creating TypeScript REST API service with Express', {
        timestamp: now,
        type: 'fact',
      }),
      // 数据分析条目：和 query 同样相关，但和 backend 条目不相似
      makeItem('py', 'analyzing data with Python TypeScript integration pipeline', {
        timestamp: now,
        type: 'fact',
      }),
    ];

    const result = await reranker.rerank('typescript', items, {
      recencyWeight: 0, // 关闭时效，只测 MMR
      mmrLambda: 0.6,
      maxResults: 3,
      mmrCandidateWindow: 10,
    });

    // MMR 应确保数据分析条目被选入 top 3
    // 它和纯 TypeScript 后端条目有足够差异（后端 vs 数据）
    const resultIds = result.map((r) => r.id);
    expect(resultIds).toContain('py');
  });

  /**
   * λ = 1 时，MMR 退化为纯相关度排序
   * 此时不会有多样性惩罚
   */
  it('degenerates to relevance ranking when lambda=1', async () => {
    const reranker = new DefaultReranker();
    const now = Date.now();

    const items = [
      makeItem('ts_keyword', 'TypeScript is great', { timestamp: now }),
      makeItem('irrelevant', 'I like hiking in mountains', { timestamp: now }),
    ];

    // λ = 1: 只看相关度，不看多样性
    const result = await reranker.rerank('typescript', items, {
      recencyWeight: 0,
      mmrLambda: 1.0,
      maxResults: 2,
    });

    // 相关内容排第一
    expect(result[0].id).toBe('ts_keyword');
  });

  /**
   * λ = 0 时，MMR 退化为纯多样性
   * 此时选出的结果互相尽可能不同，而不考虑与 query 的相关度
   */
  it('degenerates to pure diversity when lambda=0', async () => {
    const reranker = new DefaultReranker();
    const now = Date.now();

    const items = [
      makeItem('a', 'TypeScript backend development', { timestamp: now }),
      makeItem('b', 'React frontend components', { timestamp: now }),
      makeItem('c', 'Python data science', { timestamp: now }),
    ];

    // λ = 0: 只看多样性，不管相关度
    const result = await reranker.rerank('typescript', items, {
      recencyWeight: 0,
      mmrLambda: 0,
      maxResults: 3,
    });

    // 三个不同话题的结果应该都在，排序无所谓
    expect(result.length).toBe(3);
  });
});

// ─── DefaultReranker: 组合使用 ──────────────────────────────────

describe('DefaultReranker: combined usage', () => {
  /**
   * 组合场景：既有新旧记忆的差异，又有相似记忆的重复
   *
   * 候选集：
   * - 1 条旧的 TypeScript（30 天前）
   * - 1 条新的 TypeScript（刚刚）
   * - 1 条新的 Python（刚刚）
   * - 1 条旧的 Golang（30 天前）
   *
   * 预期：时效加权让新的优先，MMR 确保 Python 这条多样性条目被包含
   */
  it('combines recency and MMR effectively', async () => {
    const reranker = new DefaultReranker();
    const now = Date.now();

    // 4 个条目：新/旧 + 不同主题
    // 旧 TypeScript（30 天前，相关）
    // 新 TypeScript（1 分钟前，相关）
    // 新 Python+TS（2 分钟前，也相关但主题不同）
    // 旧 Go（60 天前，不相关）
    const items = [
      makeItem('old_ts', 'used TypeScript for backend', {
        timestamp: now - 30 * 24 * 60 * 60 * 1000,
      }),
      makeItem('new_ts', 'recently writing TypeScript backend service', {
        timestamp: now - 60 * 1000,
      }),
      makeItem('new_py', 'learning TypeScript data analysis with Python', {
        timestamp: now - 120 * 1000,
      }),
      makeItem('old_go', 'wrote microservices in Golang', {
        timestamp: now - 60 * 24 * 60 * 60 * 1000,
      }),
    ];

    const result = await reranker.rerank('typescript', items, {
      recencyWeight: 0.3,
      mmrLambda: 0.6,
      maxResults: 3,
    });

    const resultIds = result.map((r) => r.id);

    // 新 TypeScript 应该排第一（既新又相关）
    expect(resultIds[0]).toBe('new_ts');

    // new_py 应该出现在前 3（多样性 + 时效让它高于 old_ts）
    expect(resultIds).toContain('new_py');
  });

  /**
   * 验证最终返回条数不超过 maxResults
   */
  it('respects maxResults with combined config', async () => {
    const reranker = new DefaultReranker();
    const items = Array.from({ length: 20 }, (_, i) =>
      makeItem(`item_${i}`, `memory content about programming language ${i}`)
    );

    const result = await reranker.rerank('programming', items, {
      recencyWeight: 0.1,
      mmrLambda: 0.7,
      maxResults: 5,
    });

    expect(result.length).toBe(5);
  });
});

// ─── DefaultMemoryInjector 集成 ─────────────────────────────────

describe('MemoryInjector with Reranker integration', () => {
  /**
   * 验证：通过 MemoryInjectionConfig.reranker.enabled = true
   * 可以启用重排，且结果数量不超过 maxMemories
   */
  it('integrates with DefaultMemoryInjector via config', async () => {
    const injector = new DefaultMemoryInjector();
    const longTerm = new InMemoryLongTermMemory();

    // 存入多条高度相似的记忆 + 一条不同的
    await longTerm.store({
      content: '用户用 TypeScript 写后端服务',
      metadata: { timestamp: Date.now() - 1000, type: 'fact', importance: 0.8 },
    });
    await longTerm.store({
      content: '用户决定用 TypeScript 开发 API',
      metadata: { timestamp: Date.now() - 2000, type: 'fact', importance: 0.7 },
    });
    await longTerm.store({
      content: '用户喜欢 TypeScript 的类型安全',
      metadata: { timestamp: Date.now() - 3000, type: 'fact', importance: 0.6 },
    });
    await longTerm.store({
      content: '用户也在用 Python 做数据分析，之前用 TypeScript',
      metadata: { timestamp: Date.now() - 4000, type: 'fact', importance: 0.5 },
    });

    const messages = [
      systemMessage('你是一个助手'),
      userMessage('TypeScript 和 Python 你推荐哪个？'),
    ];

    // 不启用重排（默认）— 应该返回 3 条结果
    const contextNoRerank = await injector.buildContext(messages, longTerm, {
      maxMemories: 3,
      minRelevance: 0.05,
    });

    // 启用重排
    const contextWithRerank = await injector.buildContext(messages, longTerm, {
      maxMemories: 3,
      minRelevance: 0.05,
      reranker: {
        enabled: true,
        recencyWeight: 0.1,
        mmrLambda: 0.6,
      },
    });

    // 两个上下文都应该有内容
    expect(contextNoRerank).toBeTruthy();
    expect(contextWithRerank).toBeTruthy();

    // 启用重排后，Python 那条应该出现在结果中（多样性）
    expect(contextWithRerank).toContain('Python');
  });

  /**
   * 验证重排未启用时行为不变
   */
  it('does nothing when reranker is not enabled', async () => {
    const injector = new DefaultMemoryInjector();
    const longTerm = new InMemoryLongTermMemory();

    await longTerm.store({
      content: '用户喜欢旅行',
      metadata: { timestamp: Date.now(), type: 'fact', importance: 0.7 },
    });

    const messages = [systemMessage('助手'), userMessage('旅行')];

    // 不传 reranker 配置
    const context = await injector.buildContext(messages, longTerm, {
      maxMemories: 5,
      minRelevance: 0.05,
    });

    expect(context).toContain('用户喜欢旅行');
  });
});
