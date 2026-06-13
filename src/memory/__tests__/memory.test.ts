/**
 * 记忆模块测试
 *
 * 覆盖：
 * - InMemoryLongTermMemory 的 store / search / get / delete / forget
 * - TF-IDF 风格检索
 * - SimpleConsolidationStrategy 的事实提取
 * - DefaultMemorySystem 的完整流程
 */

import { describe, it, expect } from 'vitest';
import { InMemoryLongTermMemory } from '../long-term-memory.js';
import { SimpleConsolidationStrategy } from '../consolidation.js';
import { DefaultMemorySystem } from '../default-memory-system.js';
import {
  userMessage,
  assistantMessage,
} from '../../core/message.js';

// ─── InMemoryLongTermMemory ──────────────────────────────────────

describe('InMemoryLongTermMemory', () => {
  it('stores and retrieves items', async () => {
    const mem = new InMemoryLongTermMemory();

    const id = await mem.store({
      content: '用户喜欢吃辣',
      metadata: {
        timestamp: Date.now(),
        type: 'user_preference',
        tags: ['food', 'preference'],
        importance: 0.8,
      },
    });

    expect(id).toBeTruthy();
    expect(typeof id).toBe('string');

    const retrieved = await mem.get(id);
    expect(retrieved).not.toBeNull();
    expect(retrieved!.content).toBe('用户喜欢吃辣');
    expect(retrieved!.metadata.type).toBe('user_preference');
  });

  it('returns null for non-existent items', async () => {
    const mem = new InMemoryLongTermMemory();
    const result = await mem.get('non_existent');
    expect(result).toBeNull();
  });

  it('deletes items', async () => {
    const mem = new InMemoryLongTermMemory();

    const id = await mem.store({
      content: 'temp fact',
      metadata: { timestamp: Date.now(), type: 'fact' },
    });

    const deleted = await mem.delete(id);
    expect(deleted).toBe(true);

    const retrieved = await mem.get(id);
    expect(retrieved).toBeNull();
  });

  it('deleting non-existent items returns false', async () => {
    const mem = new InMemoryLongTermMemory();
    expect(await mem.delete('non_existent')).toBe(false);
  });

  it('searches by keyword relevance', async () => {
    const mem = new InMemoryLongTermMemory();

    // 只有 "喜欢" 和 "编程" 作为明确的关键词
    await mem.store({
      content: '用户喜欢吃辣的食物',
      metadata: { timestamp: 1000, type: 'user_preference', tags: ['food'], importance: 0.8 },
    });
    await mem.store({
      content: '用户喜欢编程和算法',
      metadata: { timestamp: 2000, type: 'fact', tags: ['tech', 'coding'], importance: 0.6 },
    });
    await mem.store({
      content: '用户用 Python 写后端服务',
      metadata: { timestamp: 3000, type: 'fact', tags: ['tech'], importance: 0.5 },
    });

    // 搜索 "编程" —— 应该返回 1 条（包含 "编程" 的）
    const results2 = await mem.search({ query: '编程', maxResults: 10 });
    expect(results2.length).toBe(1);
    expect(results2[0].content).toContain('编程');

    // 搜索 "Python 后端" —— 应该返回 1 条
    const results3 = await mem.search({ query: 'Python 后端', maxResults: 10 });
    expect(results3.length).toBe(1);
    expect(results3[0].content).toContain('Python');
  });

  it('supports tag filtering', async () => {
    const mem = new InMemoryLongTermMemory();

    await mem.store({
      content: '用户是学生',
      metadata: { timestamp: 1000, type: 'fact', tags: ['user'], importance: 0.5 },
    });
    await mem.store({
      content: 'Python 是主要语言',
      metadata: { timestamp: 2000, type: 'fact', tags: ['tech', 'user'], importance: 0.6 },
    });

    // 只过滤 tech 标签
    const results = await mem.search({
      query: '',
      maxResults: 10,
      tagFilter: ['tech'],
    });
    expect(results.length).toBe(1);
    expect(results[0].content).toContain('Python');
  });

  it('returns recent items when query is empty', async () => {
    const mem = new InMemoryLongTermMemory();

    await mem.store({
      content: 'first',
      metadata: { timestamp: 100, type: 'fact' },
    });
    await mem.store({
      content: 'second',
      metadata: { timestamp: 200, type: 'fact' },
    });
    await mem.store({
      content: 'third',
      metadata: { timestamp: 300, type: 'fact' },
    });

    const results = await mem.search({ query: '', maxResults: 2 });
    expect(results.length).toBe(2);
    // 最新的在前
    expect(results[0].content).toBe('third');
    expect(results[1].content).toBe('second');
  });

  it('forgets items below importance threshold', async () => {
    const mem = new InMemoryLongTermMemory();

    await mem.store({
      content: 'important fact',
      metadata: { timestamp: 100, type: 'fact', importance: 0.9 },
    });
    await mem.store({
      content: 'trivial detail',
      metadata: { timestamp: 200, type: 'fact', importance: 0.1 },
    });
    await mem.store({
      content: 'unimportant note',
      metadata: { timestamp: 300, type: 'fact', importance: 0.05 },
    });

    const removed = await mem.forget(0.5);
    expect(removed).toBe(2); // trivial detail + unimportant note

    expect(mem.size).toBe(1);
  });

  it('respects minRelevance threshold', async () => {
    const mem = new InMemoryLongTermMemory();

    await mem.store({
      content: '用户喜欢打篮球运动',
      metadata: { timestamp: 100, type: 'fact', importance: 0.5 },
    });
    await mem.store({
      content: '今天天气真的非常不错',
      metadata: { timestamp: 200, type: 'fact', importance: 0.3 },
    });

    // 低阈值能找到
    const loose = await mem.search({ query: '篮球', maxResults: 10, minRelevance: 0.01 });
    expect(loose.length).toBe(1);

    // 高阈值应该过滤掉（"天气" 对不相关 query 的分数极低）
    const strict = await mem.search({ query: '完全不相关内容', maxResults: 10, minRelevance: 0.9 });
    expect(strict.length).toBe(0);
  });

  it('clear removes everything', async () => {
    const mem = new InMemoryLongTermMemory();
    await mem.store({
      content: 'something',
      metadata: { timestamp: Date.now(), type: 'fact' },
    });
    expect(mem.size).toBe(1);
    mem.clear();
    expect(mem.size).toBe(0);
  });
});

// ─── SimpleConsolidationStrategy ─────────────────────────────────

describe('SimpleConsolidationStrategy', () => {
  it('extracts facts from user messages', () => {
    const strategy = new SimpleConsolidationStrategy();
    const messages = [
      userMessage('我喜欢吃辣的川菜，非常喜欢'),
      assistantMessage('好的，我记住了'),
      userMessage('我平时用 Python 写代码'),
    ];

    const facts = strategy.extract(messages);

    // 两条超过 10 个字符的用户消息都应该被提取
    expect(facts.length).toBe(2);

    // 第一条是 preference
    const prefFacts = facts.filter((f) => f.type === 'preference');
    expect(prefFacts.length).toBeGreaterThanOrEqual(1);
    // 至少有一条 preference 内容包含 '喜欢'
    expect(prefFacts.some((f) => f.content.includes('喜欢'))).toBe(true);
  });

  it('skips short messages (greetings)', () => {
    const strategy = new SimpleConsolidationStrategy();
    const messages = [
      userMessage('Hi'),
      userMessage('你好'),
      userMessage('我今年大三，学软件工程的'),
    ];

    const facts = strategy.extract(messages);
    // 只有第三条被提取（前两条太短）
    expect(facts.length).toBe(1);
    expect(facts[0].content).toContain('大三');
  });

  it("assigns higher importance to repeated topics", () => {
    const strategy = new SimpleConsolidationStrategy();
    const messages1 = [
      userMessage('我平时使用 TypeScript 写前端'),
    ];
    const messages2 = [
      userMessage('TypeScript 的类型系统非常好用'),
    ];
    const messages3 = [
      userMessage('我非常喜欢 TypeScript 的类型推断'),
    ];

    const facts1 = strategy.extract(messages1);
    const facts2 = strategy.extract(messages2);
    const facts3 = strategy.extract(messages3);

    // TypeScript 被提到了 3 次，最后一次的重要性应该比第一次高
    expect(facts3[0].importance).toBeGreaterThan(facts1[0].importance);
  });

  it('respects maxFactsPerConsolidation', () => {
    const strategy = new SimpleConsolidationStrategy({
      maxFactsPerConsolidation: 1,
    });

    const messages = [
      userMessage('我今天去公园玩得很开心'),
      userMessage('今天的天气非常好很适合出门'),
      userMessage('路上看到一只可爱的流浪猫'),
    ];

    const facts = strategy.extract(messages);
    expect(facts.length).toBeLessThanOrEqual(1);
  });

  it('classifies preference messages correctly', () => {
    const strategy = new SimpleConsolidationStrategy();
    const messages = [
      userMessage('我平时习惯用 Vim 编辑器写代码'),
    ];

    const facts = strategy.extract(messages);
    expect(facts.length).toBe(1);
    expect(facts[0].type).toBe('preference');
  });
});

// ─── DefaultMemorySystem ─────────────────────────────────────────

describe('DefaultMemorySystem', () => {
  it('has short-term and long-term memories', () => {
    const mem = new DefaultMemorySystem();
    expect(mem.shortTerm).toBeDefined();
    expect(mem.longTerm).toBeDefined();
  });

  it('adds messages to short-term memory', () => {
    const mem = new DefaultMemorySystem();
    mem.shortTerm.add(userMessage('Hello'));
    const messages = mem.shortTerm.getMessages();
    expect(messages.length).toBe(1);
    expect(messages[0].content).toBe('Hello');
  });

  it('consolidates short-term to long-term', async () => {
    const mem = new DefaultMemorySystem();

    mem.shortTerm.add(userMessage('我喜欢吃辣的川菜，非常喜欢'));
    mem.shortTerm.add(assistantMessage('好的，我记下了'));

    await mem.consolidate();

    // 长期记忆中应该有一条 preference
    const results = await mem.longTerm.search({
      query: '喜欢',
      maxResults: 10,
    });
    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results[0].metadata.type).toBe('user_preference');
  });

  it('queryRelevant combines short and long term', async () => {
    const mem = new DefaultMemorySystem();

    mem.shortTerm.add(userMessage('今天天气如何？'));

    await mem.longTerm.store({
      content: '用户住在北京',
      metadata: {
        timestamp: Date.now(),
        type: 'fact',
        tags: ['location'],
        importance: 0.8,
      },
    });

    const result = await mem.queryRelevant('天气');

    expect(result.recentMessages.length).toBeGreaterThanOrEqual(1);
    expect(result.relevantMemories).toBeDefined();
  });

  it('performs auto-consolidation when configured', async () => {
    const mem = new DefaultMemorySystem({ autoConsolidateInterval: 3 });

    mem.addAndMaybeConsolidate(userMessage('第一条消息不算太长'));
    mem.addAndMaybeConsolidate(userMessage('第二条消息也还可以'));

    // 还没到阈值，不触发合并
    await new Promise((r) => setTimeout(r, 100));
    let results = await mem.longTerm.search({ query: '', maxResults: 100 });
    expect(results.length).toBe(0);

    // 添加第 3 条消息，触发 auto-consolidate
    mem.addAndMaybeConsolidate(userMessage('我非常喜欢 TypeScript 语言'));
    await new Promise((r) => setTimeout(r, 100));

    results = await mem.longTerm.search({ query: '', maxResults: 100 });
    expect(results.length).toBeGreaterThanOrEqual(1);
  });
});
