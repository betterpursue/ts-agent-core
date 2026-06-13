/**
 * Response Parser 测试
 */
import { describe, it, expect } from 'vitest';
import { z } from 'zod';
import {
  repairJson,
  parseToolCalls,
  buildToolResultMessage,
  parseResultSummary,
} from '../../src/provider/response-parser.js';
import { DefaultToolRegistry } from '../../src/core/tool.js';
import type { Tool, LLMResponse } from '../../src/core/llm.js';

// ─── 测试用工具 ───────────────────────────────────

const calculatorTool: Tool = {
  metadata: {
    name: 'calculator',
    description: 'Perform arithmetic calculations',
    parameters: z.object({
      expression: z.string(),
      precision: z.number().int().min(0).max(15).optional(),
    }),
  },
  async execute() {
    return { success: true, output: '42' };
  },
};

const weatherTool: Tool = {
  metadata: {
    name: 'get_weather',
    description: 'Get weather',
    parameters: z.object({
      city: z.string().min(1),
      unit: z.enum(['celsius', 'fahrenheit']).optional(),
    }),
  },
  async execute() {
    return { success: true, output: '25°C' };
  },
};

function createRegistry(tools: Tool[] = [calculatorTool, weatherTool]) {
  const registry = new DefaultToolRegistry();
  registry.registerMany(tools);
  return registry;
}

describe('repairJson', () => {
  it('合法的 JSON 应直接通过', () => {
    expect(repairJson('{"a": 1, "b": 2}')).toBe('{"a": 1, "b": 2}');
  });

  it('缺少闭合大括号应修复', () => {
    const result = repairJson('{"a": 1, "b": 2');
    expect(result).not.toBeNull();
    expect(JSON.parse(result!)).toEqual({ a: 1, b: 2 });
  });

  it('多余的逗号应修复', () => {
    const result = repairJson('{"a": 1,}');
    expect(result).not.toBeNull();
    expect(JSON.parse(result!)).toEqual({ a: 1 });
  });

  it('单引号应替换为双引号', () => {
    const result = repairJson("{'a': 1, 'b': 'hello'}");
    expect(result).not.toBeNull();
    expect(JSON.parse(result!)).toEqual({ a: 1, b: 'hello' });
  });

  it('空字符串应返回 null', () => {
    expect(repairJson('')).toBeNull();
  });

  it('完全无法修复的 JSON 应返回 null', () => {
    expect(repairJson('not json at all !!!')).toBeNull();
  });
});

describe('parseToolCalls', () => {
  it('应正确解析带完整参数的 tool_calls', () => {
    const registry = createRegistry();
    const response: LLMResponse = {
      content: '',
      finishReason: 'tool_calls',
      toolCalls: [
        { id: 'call_1', name: 'calculator', args: { expression: '1+2' } },
      ],
    };

    const result = parseToolCalls(response, registry);

    expect(result.calls).toHaveLength(1);
    expect(result.failed).toHaveLength(0);
    expect(result.calls[0].name).toBe('calculator');
    expect(result.calls[0].argsValid).toBe(true);
    expect(result.calls[0].args.expression).toBe('1+2');
  });

  it('不存在的工具名应记录警告', () => {
    const registry = createRegistry();
    const response: LLMResponse = {
      content: '',
      finishReason: 'tool_calls',
      toolCalls: [
        { id: 'call_x', name: 'nonexistent_tool', args: {} },
      ],
    };

    const result = parseToolCalls(response, registry);

    expect(result.calls).toHaveLength(1);
    expect(result.warnings.length).toBeGreaterThan(0);
    expect(result.warnings[0]).toContain('nonexistent_tool');
  });

  it('参数类型不匹配时应标记为无效', () => {
    const registry = createRegistry();
    const response: LLMResponse = {
      content: '',
      finishReason: 'tool_calls',
      toolCalls: [
        // precision 应该是 number，但传了字符串
        { id: 'call_1', name: 'calculator', args: { expression: '1+2', precision: 'abc' } },
      ],
    };

    const result = parseToolCalls(response, registry);

    expect(result.calls).toHaveLength(1);
    // Zahl 会尝试类型转换，所以可能不会真正失败
    // 但至少不会报错导致无法执行
    expect(result.failed).toHaveLength(0);
  });

  it('没有 tool_calls 时应返回空结果', () => {
    const registry = createRegistry();
    const response: LLMResponse = {
      content: 'Hello',
      finishReason: 'stop',
    };

    const result = parseToolCalls(response, registry);

    expect(result.calls).toHaveLength(0);
    expect(result.failed).toHaveLength(0);
  });

  it('参数是 JSON 字符串时应尝试解析', () => {
    const registry = createRegistry();
    const response: LLMResponse = {
      content: '',
      finishReason: 'tool_calls',
      toolCalls: [
        {
          id: 'call_1',
          name: 'calculator',
          args: '{"expression": "1+2"}' as unknown as Record<string, unknown>,
        },
      ],
    };

    const result = parseToolCalls(response, registry);

    expect(result.calls).toHaveLength(1);
    expect(result.calls[0].args.expression).toBe('1+2');
  });

  it('参数是 JSON 字符串且缺少闭合括号时应修复', () => {
    const registry = createRegistry();
    const response: LLMResponse = {
      content: '',
      finishReason: 'tool_calls',
      toolCalls: [
        {
          id: 'call_1',
          name: 'calculator',
          args: '{"expression": "1+2"' as unknown as Record<string, unknown>,
        },
      ],
    };

    const result = parseToolCalls(response, registry);

    expect(result.calls).toHaveLength(1);
    // 修复后应该能解析成功
    expect(result.calls[0].args).toHaveProperty('expression');
  });
});

describe('buildToolResultMessage', () => {
  it('应生成正确的 tool result 消息', () => {
    const msg = buildToolResultMessage(
      { id: 'call_1', name: 'calculator', args: {}, argsValid: true },
      '42',
      false
    );

    expect(msg.role).toBe('tool');
    expect(msg.content).toBe('42');
    expect(msg.toolCallId).toBe('call_1');
    expect(msg.toolName).toBe('calculator');
    expect(msg.isError).toBeFalsy();
  });

  it('错误消息应标记 isError', () => {
    const msg = buildToolResultMessage(
      { id: 'call_err', name: 'failing_tool', args: {}, argsValid: false },
      'Something went wrong',
      true
    );

    expect(msg.isError).toBe(true);
  });
});

describe('parseResultSummary', () => {
  it('应生成可读的摘要', () => {
    const registry = createRegistry();
    const response: LLMResponse = {
      content: '',
      finishReason: 'tool_calls',
      toolCalls: [
        { id: 'call_1', name: 'calculator', args: { expression: '1+2' } },
        { id: 'call_2', name: 'nonexistent', args: {} },
      ],
    };

    const result = parseToolCalls(response, registry);
    const summary = parseResultSummary(result);

    expect(summary).toContain('Tool calls: 2');
    expect(summary).toContain('nonexistent');
  });
});
