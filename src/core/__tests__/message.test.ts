/**
 * 消息模块测试
 *
 * 覆盖：
 * - 消息工厂函数
 * - 消息校验（单条 + 序列）
 * - Token 估算
 * - 消息窗口管理（添加、裁剪、策略）
 * - Provider 适配器
 */

import { describe, it, expect } from 'vitest';
import {
  systemMessage,
  userMessage,
  assistantMessage,
  toolResultMessage,
  validateMessage,
  validateConversation,
  TokenEstimator,
  MessageWindow,
} from '../message.js';

import {
  toOpenAIMessages,
  fromOpenAIToolCalls,
  toAnthropicMessages,
  extractAnthropicSystemPrompt,
} from '../provider-adapter.js';

// ─── 工厂函数 ────────────────────────────────────────────────────

describe('message factory functions', () => {
  it('systemMessage creates a system message', () => {
    const msg = systemMessage('You are a helpful assistant.');
    expect(msg.role).toBe('system');
    expect(msg.content).toBe('You are a helpful assistant.');
  });

  it('userMessage creates a user message', () => {
    const msg = userMessage('Hello!');
    expect(msg.role).toBe('user');
    expect(msg.content).toBe('Hello!');
  });

  it('assistantMessage creates an assistant message without tool calls', () => {
    const msg = assistantMessage('Hi there!');
    expect(msg.role).toBe('assistant');
    expect(msg.content).toBe('Hi there!');
    expect(msg.toolCalls).toBeUndefined();
  });

  it('assistantMessage creates an assistant message with tool calls', () => {
    const msg = assistantMessage('', [
      { id: 'call_1', name: 'calculator', args: { expression: '1+1' } },
    ]);
    expect(msg.role).toBe('assistant');
    expect(msg.toolCalls).toHaveLength(1);
    expect(msg.toolCalls![0].name).toBe('calculator');
  });

  it('toolResultMessage creates a tool result message', () => {
    const msg = toolResultMessage('call_1', 'calculator', '2');
    expect(msg.role).toBe('tool');
    expect(msg.toolCallId).toBe('call_1');
    expect(msg.toolName).toBe('calculator');
    expect(msg.isError).toBeUndefined();
  });

  it('toolResultMessage marks error when isError is true', () => {
    const msg = toolResultMessage('call_1', 'calculator', 'Error: invalid input', true);
    expect(msg.role).toBe('tool');
    expect(msg.isError).toBe(true);
  });
});

// ─── 消息校验 ────────────────────────────────────────────────────

describe('validateMessage', () => {
  it('accepts a valid system message', () => {
    const result = validateMessage(systemMessage('Be helpful.'));
    expect(result.valid).toBe(true);
    expect(result.errors).toHaveLength(0);
  });

  it('accepts a valid assistant message with toolCalls', () => {
    const msg = assistantMessage('', [
      { id: 'tc_1', name: 'calc', args: { x: 1 } },
    ]);
    const result = validateMessage(msg);
    expect(result.valid).toBe(true);
  });

  it('rejects a message without role', () => {
    const result = validateMessage({ role: '' as any, content: 'hi' });
    expect(result.valid).toBe(false);
  });

  it('rejects a tool message without toolCallId', () => {
    const msg = { role: 'tool' as const, content: 'result', toolName: 'calc' };
    const result = validateMessage(msg);
    expect(result.valid).toBe(false);
    expect(result.errors[0]).toContain('toolCallId');
  });

  it('rejects a user message with toolCalls', () => {
    const msg = {
      role: 'user' as const,
      content: 'hi',
      toolCalls: [{ id: 'tc_1', name: 'calc', args: {} }],
    };
    const result = validateMessage(msg);
    expect(result.valid).toBe(false);
    expect(result.errors[0]).toContain('Only assistant');
  });

  it('rejects an assistant message with toolCallId', () => {
    const msg = { role: 'assistant' as const, content: 'hi', toolCallId: 'tc_1' };
    const result = validateMessage(msg);
    expect(result.valid).toBe(false);
    expect(result.errors[0]).toContain('should not have toolCallId');
  });

  it('rejects a toolCalls entry without id', () => {
    const msg = assistantMessage('', [
      { id: '', name: 'calc', args: {} },
    ]);
    const result = validateMessage(msg);
    expect(result.valid).toBe(false);
    expect(result.errors[0]).toContain('id');
  });
});

describe('validateConversation', () => {
  it('accepts a valid conversation', () => {
    const msgs = [
      systemMessage('You are helpful.'),
      userMessage('Hi'),
      assistantMessage('Hello!'),
    ];
    const result = validateConversation(msgs);
    expect(result.valid).toBe(true);
  });

  it('accepts a conversation with tool calls', () => {
    const msgs = [
      systemMessage('Be helpful.'),
      userMessage('What is 1+1?'),
      assistantMessage('', [
        { id: 'tc_1', name: 'calculator', args: { expression: '1+1' } },
      ]),
      toolResultMessage('tc_1', 'calculator', '2'),
      assistantMessage('The answer is 2.'),
    ];
    const result = validateConversation(msgs);
    expect(result.valid).toBe(true);
  });

  it('rejects empty conversation', () => {
    const result = validateConversation([]);
    expect(result.valid).toBe(false);
  });

  it('rejects consecutive user messages without tool', () => {
    const msgs = [userMessage('Hi'), userMessage('Hello?')];
    const result = validateConversation(msgs);
    expect(result.valid).toBe(false);
    expect(result.errors[0]).toContain('consecutive');
  });

  it('rejects tool message without matching tool_call', () => {
    const msgs = [
      userMessage('Hi'),
      assistantMessage('Hello!'),
      toolResultMessage('tc_1', 'calc', '2'),
    ];
    const result = validateConversation(msgs);
    expect(result.valid).toBe(false);
    expect(result.errors[0]).toContain('no matching');
  });
});

// ─── Token 估算 ──────────────────────────────────────────────────

describe('TokenEstimator', () => {
  const estimator = new TokenEstimator();

  it('estimates 0 tokens for empty text', () => {
    expect(estimator.estimateText('')).toBe(0);
  });

  it('estimates simple English text', () => {
    const tokens = estimator.estimateText('hello world');
    expect(tokens).toBeGreaterThan(0);
    expect(tokens).toBeLessThan(10);
  });

  it('estimates Chinese text with higher weight', () => {
    const englishTokens = estimator.estimateText('abcdefgh');
    const chineseTokens = estimator.estimateText('你好世界');
    // Chinese chars have higher weight
    expect(chineseTokens).toBeGreaterThan(englishTokens);
  });

  it('estimates a complete message', () => {
    const msg = systemMessage('You are a helpful assistant.');
    const tokens = estimator.estimateMessage(msg);
    expect(tokens).toBeGreaterThan(0);
  });

  it('estimates messages with tool calls', () => {
    const msg = assistantMessage('', [
      { id: 'tc_1', name: 'calculator', args: { expression: '2+2' } },
    ]);
    const tokens = estimator.estimateMessage(msg);
    expect(tokens).toBeGreaterThan(0);
  });

  it('estimates a batch of messages', () => {
    const msgs = [
      systemMessage('Be helpful.'),
      userMessage('What is AI?'),
      assistantMessage('AI is...'),
    ];
    const total = estimator.estimateMessages(msgs);
    expect(total).toBeGreaterThan(0);
  });
});

// ─── 消息窗口 ────────────────────────────────────────────────────

describe('MessageWindow', () => {
  it('starts empty', () => {
    const w = new MessageWindow();
    expect(w.isEmpty).toBe(true);
    expect(w.length).toBe(0);
  });

  it('adds messages', () => {
    const w = new MessageWindow();
    w.add(systemMessage('Be helpful.'));
    w.add(userMessage('Hi'));
    expect(w.length).toBe(2);
    expect(w.isEmpty).toBe(false);
  });

  it('clears messages', () => {
    const w = new MessageWindow();
    w.add(userMessage('Hi'));
    w.clear();
    expect(w.isEmpty).toBe(true);
  });

  it('returns a copy of messages', () => {
    const w = new MessageWindow();
    w.add(userMessage('Hi'));
    const msgs = w.getAll();
    msgs.push(userMessage('Hey'));
    expect(w.length).toBe(1); // original unchanged
  });

  it('trims oldest pair when over budget', () => {
    const w = new MessageWindow({
      maxTokens: 100, // very small budget
      trimStrategy: 'drop_oldest_pair',
    });

    w.add(systemMessage('Be helpful.')); // always kept
    w.add(userMessage('What is 1+1?'));
    w.add(assistantMessage('The answer is 2.'));

    // system should always be kept; oldest user/assistant pair may be trimmed
    const msgs = w.getAll();
    expect(msgs.length).toBeGreaterThanOrEqual(1);
    expect(msgs[0].role).toBe('system');
  });

  it('keeps recent messages when trimming', () => {
    const w = new MessageWindow({
      maxTokens: 200,
      trimStrategy: 'drop_oldest_pair',
    });

    // Add system + several rounds of conversation
    w.add(systemMessage('Be helpful.'));
    w.add(userMessage('Q1'));
    w.add(assistantMessage('A1'));
    w.add(userMessage('Q2'));
    w.add(assistantMessage('A2'));
    w.add(userMessage('Q3'));
    w.add(assistantMessage('A3'));

    const msgs = w.getAll();
    // The last user/assistant pair should be preserved
    const lastAssistant = msgs[msgs.length - 1];
    expect(lastAssistant.role).toBe('assistant');
    expect(lastAssistant.content).toBe('A3');

    // system prompt should be preserved
    expect(msgs[0].role).toBe('system');
  });

  it('removes tool results when their assistant is dropped (drop_oldest)', () => {
    const w = new MessageWindow({
      maxTokens: 80,
      trimStrategy: 'drop_oldest',
    });

    w.add(systemMessage('Sys.'));
    w.add(userMessage('What is 2+2?'));
    w.add(
      assistantMessage('', [
        { id: 'tc_1', name: 'calculator', args: { expression: '2+2' } },
      ])
    );
    w.add(toolResultMessage('tc_1', 'calculator', '4'));
    w.add(assistantMessage('4 is the answer.'));

    const msgs = w.getAll();
    // Check no tool result without its corresponding assistant
    for (const m of msgs) {
      if (m.role === 'tool') {
        const hasAssistant = msgs.some(
          (am) =>
            am.role === 'assistant' &&
            am.toolCalls?.some((tc) => tc.id === m.toolCallId)
        );
        expect(hasAssistant).toBe(true);
      }
    }
  });
});

// ─── Provider 适配器 ─────────────────────────────────────────────

describe('toOpenAIMessages', () => {
  it('converts a simple conversation', () => {
    const msgs = [
      systemMessage('Be helpful.'),
      userMessage('Hi'),
      assistantMessage('Hello!'),
    ];
    const result = toOpenAIMessages(msgs);
    expect(result).toHaveLength(3);
    expect(result[0].role).toBe('system');
    expect(result[1].role).toBe('user');
    expect(result[2].role).toBe('assistant');
  });

  it('converts tool_calls to OpenAI format', () => {
    const msgs = [
      userMessage('What is 2+2?'),
      assistantMessage('', [
        { id: 'call_abc', name: 'calculator', args: { expression: '2+2' } },
      ]),
    ];
    const result = toOpenAIMessages(msgs);
    const assistantMsg = result[1];
    expect(assistantMsg.tool_calls).toHaveLength(1);
    expect(assistantMsg.tool_calls![0].function.name).toBe('calculator');
    expect(assistantMsg.tool_calls![0].function.arguments).toBe(
      '{"expression":"2+2"}'
    );
  });

  it('converts tool result messages', () => {
    const msgs = [
      userMessage('What is 2+2?'),
      assistantMessage('', [
        { id: 'call_abc', name: 'calculator', args: { expression: '2+2' } },
      ]),
      toolResultMessage('call_abc', 'calculator', '4'),
    ];
    const result = toOpenAIMessages(msgs);
    const toolMsg = result[2];
    expect(toolMsg.role).toBe('tool');
    expect(toolMsg.tool_call_id).toBe('call_abc');
    expect(toolMsg.name).toBe('calculator');
  });
});

describe('fromOpenAIToolCalls', () => {
  it('converts OpenAI tool_calls to internal format', () => {
    const openaiToolCalls = [
      {
        id: 'call_1',
        type: 'function' as const,
        function: { name: 'calculator', arguments: '{"x":1}' },
      },
    ];
    const result = fromOpenAIToolCalls(openaiToolCalls);
    expect(result).toHaveLength(1);
    expect(result![0].name).toBe('calculator');
    expect(result![0].args).toEqual({ x: 1 });
  });

  it('returns undefined for empty tool_calls', () => {
    expect(fromOpenAIToolCalls([])).toBeUndefined();
    expect(fromOpenAIToolCalls()).toBeUndefined();
  });
});

describe('toAnthropicMessages', () => {
  it('converts user/assistant messages', () => {
    const msgs = [userMessage('Hi'), assistantMessage('Hello!')];
    const result = toAnthropicMessages(msgs);
    expect(result).toHaveLength(2);
    expect(result[0].role).toBe('user');
    expect(result[1].role).toBe('assistant');
  });

  it('separates system prompt by default', () => {
    const msgs = [
      systemMessage('Be helpful.'),
      userMessage('Hi'),
      assistantMessage('Hello!'),
    ];
    const result = toAnthropicMessages(msgs);
    // system message should not be in the result
    expect(result.length).toBe(2);
    expect(result[0].role).toBe('user');
    expect(result[1].role).toBe('assistant');
  });

  it('converts tool_use content blocks', () => {
    const msgs = [
      userMessage('What is 2+2?'),
      assistantMessage('', [
        { id: 'call_1', name: 'calculator', args: { expression: '2+2' } },
      ]),
    ];
    const result = toAnthropicMessages(msgs);
    const assistantContent = result[1].content;
    expect(assistantContent[0].type).toBe('tool_use');
    if (assistantContent[0].type === 'tool_use') {
      expect(assistantContent[0].name).toBe('calculator');
    }
  });

  it('converts tool_result content blocks', () => {
    const msgs = [
      userMessage('What is 2+2?'),
      assistantMessage('', [
        { id: 'call_1', name: 'calculator', args: { expression: '2+2' } },
      ]),
      toolResultMessage('call_1', 'calculator', '4'),
    ];
    const result = toAnthropicMessages(msgs);
    // tool_result should be in a user message (Anthropic convention)
    const userMsgs = result.filter(
      (m) =>
        m.role === 'user' &&
        m.content.some((c) => c.type === 'tool_result')
    );
    expect(userMsgs.length).toBeGreaterThan(0);
  });
});

describe('extractAnthropicSystemPrompt', () => {
  it('extracts system prompt from messages', () => {
    const msgs = [
      systemMessage('You are helpful.'),
      systemMessage('You speak Chinese.'),
      userMessage('Hi'),
    ];
    const prompt = extractAnthropicSystemPrompt(msgs);
    expect(prompt).toBe('You are helpful.\nYou speak Chinese.');
  });

  it('returns undefined when no system messages', () => {
    const msgs = [userMessage('Hi')];
    expect(extractAnthropicSystemPrompt(msgs)).toBeUndefined();
  });
});
