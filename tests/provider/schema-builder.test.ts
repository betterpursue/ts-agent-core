/**
 * Schema Builder 测试
 */
import { describe, it, expect } from 'vitest';
import { z } from 'zod';
import {
  buildOpenAITool,
  buildOpenAITools,
  buildAnthropicTool,
  validateToolSchema,
  SchemaCache,
} from '../../src/provider/schema-builder.js';
import { DefaultToolRegistry } from '../../src/core/tool.js';
import type { Tool, ToolMetadata } from '../../src/core/tool.js';

// ─── 测试用工具 ───────────────────────────────────

const calculatorMetadata: ToolMetadata = {
  name: 'calculator',
  description: 'Perform arithmetic calculations',
  parameters: z.object({
    expression: z.string().describe('The math expression to evaluate'),
    precision: z.number().int().min(0).max(15).optional(),
  }),
};

const weatherTool: Tool = {
  metadata: {
    name: 'get_weather',
    description: 'Get current weather for a city',
    parameters: z.object({
      city: z.string().describe('City name'),
      unit: z.enum(['celsius', 'fahrenheit']).optional().describe('Temperature unit'),
    }),
  },
  async execute() {
    return { success: true, output: '25°C' };
  },
};

describe('Schema Builder', () => {
  describe('buildOpenAITool', () => {
    it('应该生成正确的 OpenAI tool 结构', () => {
      const tool = buildOpenAITool(calculatorMetadata);

      expect(tool.type).toBe('function');
      expect(tool.function.name).toBe('calculator');
      expect(tool.function.description).toBe('Perform arithmetic calculations');
      expect(tool.function.parameters).toBeDefined();
      expect(tool.function.parameters).toHaveProperty('type', 'object');
    });

    it('应该正确映射参数类型和 required 字段', () => {
      const tool = buildOpenAITool(calculatorMetadata);
      const params = tool.function.parameters as Record<string, unknown>;

      expect(params).toHaveProperty('type', 'object');
      expect((params as { properties: Record<string, unknown> }).properties).toBeDefined();

      const properties = (params as { properties: Record<string, unknown> }).properties;
      expect(properties).toHaveProperty('expression');
      expect(properties).toHaveProperty('precision');

      // expression 是 required
      expect((params as { required?: string[] }).required).toContain('expression');
    });

    it('strict 模式下所有参数应 required', () => {
      const tool = buildOpenAITool(calculatorMetadata, true);
      const params = tool.function.parameters as Record<string, unknown>;

      // 所有 properties 的 key 都应在 required 中
      const required = (params as { required?: string[] }).required ?? [];
      const propKeys = Object.keys(
        (params as { properties: Record<string, unknown> }).properties
      );
      expect(propKeys.length).toBe(required.length);
    });
  });

  describe('buildOpenAITools', () => {
    it('应该批量生成 tools 数组', () => {
      const metadatas = [calculatorMetadata, weatherTool.metadata];
      const tools = buildOpenAITools(metadatas);

      expect(tools).toHaveLength(2);
      expect(tools[0].function.name).toBe('calculator');
      expect(tools[1].function.name).toBe('get_weather');
    });
  });

  describe('buildAnthropicTool', () => {
    it('应该生成 Anthropic 格式的 tool', () => {
      const tool = buildAnthropicTool(calculatorMetadata);

      expect(tool.name).toBe('calculator');
      expect(tool.description).toBe('Perform arithmetic calculations');
      expect(tool.input_schema).toBeDefined();
      expect(tool.input_schema).toHaveProperty('type', 'object');
    });
  });

  describe('validateToolSchema', () => {
    it('合法的 schema 应通过校验', () => {
      const result = validateToolSchema(calculatorMetadata);
      expect(result.valid).toBe(true);
      expect(result.errors).toHaveLength(0);
    });

    it('空的 name 应报错', () => {
      const result = validateToolSchema({
        name: '',
        description: 'test',
        parameters: z.object({}),
      });
      expect(result.valid).toBe(false);
    });

    it('空的 description 应报错', () => {
      const result = validateToolSchema({
        name: 'test',
        description: '',
        parameters: z.object({}),
      });
      expect(result.valid).toBe(false);
    });
  });

  describe('SchemaCache', () => {
    it('应该缓存 schema 转换结果', () => {
      const cache = new SchemaCache();

      const first = cache.getOrCreateOpenAI(calculatorMetadata);
      const second = cache.getOrCreateOpenAI(calculatorMetadata);

      // 同一个引用
      expect(first).toBe(second);
    });

    it('invalidate 后应重新生成', () => {
      const cache = new SchemaCache();

      const first = cache.getOrCreateOpenAI(calculatorMetadata);
      cache.invalidate();
      const second = cache.getOrCreateOpenAI(calculatorMetadata);

      expect(first).not.toBe(second);
      expect(second.function.name).toBe('calculator');
    });

    it('应该批量获取', () => {
      const cache = new SchemaCache();
      const tools = cache.getOpenAITools([calculatorMetadata, weatherTool.metadata]);

      expect(tools).toHaveLength(2);
      expect(tools[0].function.name).toBe('calculator');
      expect(tools[1].function.name).toBe('get_weather');
    });
  });
});
