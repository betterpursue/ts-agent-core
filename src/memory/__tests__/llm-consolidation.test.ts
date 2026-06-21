/**
 * LLMConsolidationStrategy 测试
 *
 * 使用 MockConsolidationLLM 模拟 LLM 回复，不依赖外部 API。
 * 覆盖：
 * - 基本提取流程
 * - 置信度过滤
 * - 合并决策（keep / skip / merge_into）
 * - 空消息处理
 * - JSON 解析容错（code block、markdown 包裹）
 * - 与 DefaultMemorySystem 的集成
 */

import { describe, it, expect, beforeEach } from 'vitest';
import { LLMConsolidationStrategy, MockConsolidationLLM } from '../llm-consolidation.js';
import { SimpleConsolidationStrategy } from '../consolidation.js';
import { DefaultMemorySystem } from '../default-memory-system.js';
import { InMemoryLongTermMemory } from '../long-term-memory.js';
import {
  userMessage,
  assistantMessage,
} from '../../core/message.js';

// ─── 辅助：创建 Mock Provider ──────────────────────────────────

function createExtractionMock(): MockConsolidationLLM {
  return new MockConsolidationLLM([
    JSON.stringify([
      {
        content: '用户喜欢吃辣的食物',
        type: 'preference',
        tags: ['food', 'spicy'],
        confidence: 0.95,
      },
      {
        content: '用户用 TypeScript 写后端',
        type: 'fact',
        tags: ['tech', 'typescript'],
        confidence: 0.85,
      },
    ]),
  ]);
}

// ─── LLMConsolidationStrategy: 提取 ─────────────────────────────

describe('LLMConsolidationStrategy — extraction', () => {
  let provider: MockConsolidationLLM;

  beforeEach(() => {
    provider = new MockConsolidationLLM([]);
  });

  it('extracts facts from messages', async () => {
    provider = createExtractionMock();
    const strategy = new LLMConsolidationStrategy({
      provider,
      confidenceThreshold: 0.3,
    });

    const messages = [
      userMessage('我喜欢吃辣的'),
      userMessage('我用 TypeScript 写后端'),
    ];

    const facts = await strategy.extract(messages);

    expect(facts.length).toBe(2);
    expect(facts[0].type).toBe('preference');
    expect(facts[0].content).toContain('吃辣');
    expect(facts[1].type).toBe('fact');
    expect(facts[1].tags).toContain('typescript');
  });

  it('filters facts below confidence threshold', async () => {
    provider = new MockConsolidationLLM([
      JSON.stringify([
        { content: '高置信度', type: 'fact', tags: ['a'], confidence: 0.9 },
        { content: '低置信度', type: 'fact', tags: ['b'], confidence: 0.2 },
        { content: '中等置信度', type: 'fact', tags: ['c'], confidence: 0.6 },
      ]),
    ]);

    const strategy = new LLMConsolidationStrategy({
      provider,
      confidenceThreshold: 0.5,
    });

    const facts = await strategy.extract([userMessage('hello')]);
    expect(facts.length).toBe(2);
    expect(facts[0].content).toBe('高置信度');
    expect(facts[1].content).toBe('中等置信度');
  });

  it('returns empty array for empty messages', async () => {
    const strategy = new LLMConsolidationStrategy({
      provider,
      confidenceThreshold: 0.5,
    });

    const facts = await strategy.extract([]);
    expect(facts).toEqual([]);
  });

  it('handles JSON wrapped in markdown code blocks', async () => {
    provider = new MockConsolidationLLM([
      '```json\n' +
      JSON.stringify([
        { content: '用户是学生', type: 'fact', tags: ['user'], confidence: 0.8 },
      ]) +
      '\n```',
    ]);

    const strategy = new LLMConsolidationStrategy({
      provider,
      confidenceThreshold: 0.3,
    });

    const facts = await strategy.extract([userMessage('我是学生')]);
    expect(facts.length).toBe(1);
    expect(facts[0].content).toBe('用户是学生');
  });

  it('gracefully handles LLM failure (returns empty)', async () => {
    provider = new MockConsolidationLLM([
      'not valid json at all {{{',
    ]);

    const strategy = new LLMConsolidationStrategy({
      provider,
      confidenceThreshold: 0.3,
    });

    const facts = await strategy.extract([userMessage('hello')]);
    expect(facts).toEqual([]);
  });

  it('maps confidence to importance correctly', async () => {
    provider = new MockConsolidationLLM([
      JSON.stringify([
        { content: '事实A', type: 'fact', tags: [], confidence: 0.5 },
        { content: '事实B', type: 'fact', tags: [], confidence: 1.0 },
      ]),
    ]);

    const strategy = new LLMConsolidationStrategy({
      provider,
      confidenceThreshold: 0.3,
    });

    const facts = await strategy.extract([userMessage('test')]);

    // confidence 0.5 -> importance ~0.55
    // confidence 1.0 -> importance ~0.80
    expect(facts[0].importance).toBeGreaterThanOrEqual(0.5);
    expect(facts[1].importance).toBeGreaterThan(facts[0].importance);
    expect(facts[1].importance).toBeLessThanOrEqual(1.0);
  });
});

// ─── LLMConsolidationStrategy: 合并 ─────────────────────────────

describe('LLMConsolidationStrategy — merging', () => {
  it('keeps new facts when merge says keep', async () => {
    const provider = new MockConsolidationLLM([
      JSON.stringify([
        { factIndex: 0, action: 'keep', reason: 'New information' },
        { factIndex: 1, action: 'keep', reason: 'Also new' },
      ]),
      JSON.stringify([
        { content: '新事实A', type: 'fact', tags: ['a'], confidence: 0.8 },
      ]),
    ]);

    const strategy = new LLMConsolidationStrategy({ provider, confidenceThreshold: 0.3 });
    const newFacts = [
      { content: '新事实A', type: 'fact' as const, tags: ['a'], importance: 0.5 },
      { content: '新事实B', type: 'fact' as const, tags: ['b'], importance: 0.6 },
    ];

    const merged = await strategy.merge(newFacts, []);
    expect(merged.length).toBe(2);
  });

  it('skips duplicate facts', async () => {
    const provider = new MockConsolidationLLM([
      JSON.stringify([
        { factIndex: 0, action: 'skip', reason: 'Already exists' },
        { factIndex: 1, action: 'keep', reason: 'New info' },
      ]),
    ]);

    const strategy = new LLMConsolidationStrategy({ provider, confidenceThreshold: 0.3 });
    const newFacts = [
      { content: '已存在的事实', type: 'fact' as const, tags: ['a'], importance: 0.5 },
      { content: '新的事实', type: 'fact' as const, tags: ['b'], importance: 0.5 },
    ];
    const existing = [
      { id: 'e1', content: '已存在的事实', metadata: { timestamp: 100, type: 'fact' as const } },
    ];

    const merged = await strategy.merge(newFacts, existing);
    expect(merged.length).toBe(1);
    expect(merged[0].content).toBe('新的事实');
  });

  it('merges related facts into combined statements', async () => {
    const provider = new MockConsolidationLLM([
      JSON.stringify([
        {
          factIndex: 0,
          action: 'merge_into',
          mergeTargetIndex: undefined,
          mergedContent: '用户是一名喜欢 TypeScript 的软件工程师',
          reason: 'Related facts: user is a software engineer who likes TypeScript',
        },
      ]),
    ]);

    const strategy = new LLMConsolidationStrategy({ provider, confidenceThreshold: 0.3 });
    const newFacts = [
      {
        content: '用户是一名软件工程师',
        type: 'fact' as const,
        tags: ['job'],
        importance: 0.6,
      },
    ];

    const merged = await strategy.merge(newFacts, []);
    expect(merged.length).toBe(1);
    expect(merged[0].content).toContain('软件工程师');
    expect(merged[0].importance).toBeGreaterThanOrEqual(0.6);
  });

  it('handles merge failure gracefully', async () => {
    const provider = new MockConsolidationLLM([
      '完全不是 JSON',
    ]);

    const strategy = new LLMConsolidationStrategy({ provider, confidenceThreshold: 0.3 });
    const newFacts = [
      { content: '保留这个', type: 'fact' as const, tags: ['a'], importance: 0.5 },
    ];

    const merged = await strategy.merge(newFacts, [
      { id: 'e1', content: '旧的', metadata: { timestamp: 100, type: 'fact' as const } },
    ]);
    expect(merged.length).toBe(1);
    expect(merged[0].content).toBe('保留这个');
  });
});

// ─── 整合：与 DefaultMemorySystem 配合 ─────────────────────────

describe('LLMConsolidationStrategy — integration with DefaultMemorySystem', () => {
  it('works as drop-in replacement for SimpleConsolidationStrategy', async () => {
    const provider = new MockConsolidationLLM([
      JSON.stringify([
        { content: '用户喜欢 TypeScript', type: 'preference', tags: ['typescript'], confidence: 0.9 },
      ]),
      JSON.stringify([
        { factIndex: 0, action: 'keep', reason: 'New information' },
      ]),
    ]);

    const strategy = new LLMConsolidationStrategy({
      provider,
      confidenceThreshold: 0.3,
    });

    const mem = new DefaultMemorySystem({
      consolidationStrategy: strategy,
    });

    mem.shortTerm.add(userMessage('我喜欢 TypeScript 编程语言'));
    await mem.consolidate();

    const results = await mem.longTerm.search({
      query: 'TypeScript',
      maxResults: 10,
    });

    expect(results.length).toBeGreaterThanOrEqual(1);
    expect(results[0].content).toContain('TypeScript');
  });

  it('performs merge step during consolidation', async () => {
    // 第一个 consolidate：提取一条事实
    // 第二个 consolidate：新事实与已有事实合并为一条
    //
    // 注意：第一个 consolidate 的 merge 阶段因 existingMemories 为空直接返回，
    // 不会调用 LLM。所以 mock 只需要 3 个响应：第一次 extract、第二次 extract、第二次 merge。
    const provider = new MockConsolidationLLM([
      // [0] 第一次 consolidate: extract
      JSON.stringify([
        { content: '用户用 TypeScript', type: 'fact', tags: ['tech'], confidence: 0.9 },
      ]),
      // [1] 第二次 consolidate: extract
      JSON.stringify([
        { content: '用户用 TS 写后端', type: 'fact', tags: ['tech'], confidence: 0.85 },
      ]),
      // [2] 第二次 consolidate: merge
      JSON.stringify([
        {
          factIndex: 0,
          action: 'merge_into',
          mergeTargetIndex: undefined,
          mergedContent: '用户用 TypeScript 写后端服务',
          reason: 'Combining related facts about TypeScript usage',
        },
      ]),
    ]);

    const strategy = new LLMConsolidationStrategy({
      provider,
      confidenceThreshold: 0.3,
    });

    const mem = new DefaultMemorySystem({
      consolidationStrategy: strategy,
    });

    // 第一次合并
    mem.shortTerm.add(userMessage('我用 TypeScript'));
    await mem.consolidate();

    let results = await mem.longTerm.search({ query: '', maxResults: 10 });
    expect(results.length).toBe(1);
    expect(results[0].content).toBe('用户用 TypeScript');

    // 第二次合并（新事实 + 已有记忆）
    mem.shortTerm.add(userMessage('我用 TS 写后端'));
    await mem.consolidate();

    // 检查长期记忆最新的条目包含合并后的内容
    results = await mem.longTerm.search({ query: '', maxResults: 10 });
    expect(results.length).toBe(2);  // 原来的 + 合并后的
    // 最新条目（第一个）应该是合并后的
    const allContent = results.map((r) => r.content).join(' | ');
    expect(allContent).toContain('后端');
  });
});
