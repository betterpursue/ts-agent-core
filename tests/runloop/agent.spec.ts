/**
 * DefaultAgent 测试
 *
 * 测试覆盖：
 * 1. 直接回答（无工具调用）
 * 2. 单次工具调用 → 最终回答
 * 3. 多次工具调用
 * 4. 达到 maxIterations 提前终止
 * 5. 工具不存在时的错误处理
 * 6. 工具执行抛出异常时的错误隔离
 * 7. 记忆系统集成（run 之间保持上下文）
 * 8. Hooks 触发
 */

import { describe, it, expect, vi } from 'vitest';
import { DefaultAgent } from '../../src/runloop/agent.js';
import { DefaultToolRegistry } from '../../src/core/tool.js';
import { SlidingWindowMemory } from '../../src/core/memory.js';
import type { MemorySystem } from '../../src/core/memory.js';
import type { LLMProvider, LLMResponse, LLMRequest } from '../../src/core/llm.js';
import type { Tool, ToolResult } from '../../src/core/tool.js';
import type { AgentConfig } from '../../src/core/agent.js';
import { z } from 'zod';

// ─── 测试辅助函数 ───────────────────────────────────

function createMockProvider(responses: LLMResponse[]): LLMProvider {
  let index = 0;
  return {
    name: 'mock',
    async complete(_request: LLMRequest): Promise<LLMResponse> {
      if (index >= responses.length) {
        return { content: 'No more responses configured', finishReason: 'stop' };
      }
      return responses[index++];
    },
    async completeStream(
      _request: LLMRequest,
      _onChunk: (chunk: any) => void
    ): Promise<LLMResponse> {
      if (index >= responses.length) {
        return { content: 'No more responses configured', finishReason: 'stop' };
      }
      return responses[index++];
    },
  };
}

/** 一个简单的计算器工具（测试用） */
const calculatorTool: Tool = {
  metadata: {
    name: 'calculator',
    description: 'Perform arithmetic calculation',
    parameters: z.object({
      expression: z.string().describe('Arithmetic expression, e.g. 1+2'),
    }),
  },
  async execute(args: Record<string, unknown>): Promise<ToolResult> {
    const expression = args['expression'] as string;
    try {
      // 安全：只测试用，且只允许数字和运算符
      const sanitized = expression.replace(/[^0-9+\-*/.()]/g, '');
      const result = Function(`"use strict"; return (${sanitized})`)();
      return { success: true, output: String(result) };
    } catch (err) {
      return {
        success: false,
        output: `Error: ${err instanceof Error ? err.message : String(err)}`,
        error: String(err),
      };
    }
  },
};

/** 创建一个完整的测试用 Agent */
function createTestAgent(options: {
  provider: LLMProvider;
  tools?: Tool[];
  systemPrompt?: string;
  maxIterations?: number;
  memory?: MemorySystem;
}): DefaultAgent {
  const toolRegistry = new DefaultToolRegistry();
  if (options.tools) {
    toolRegistry.registerMany(options.tools);
  }

  const shortTermMemory =
    options.memory?.shortTerm ?? new SlidingWindowMemory(4096);
  const longTermMemory = options.memory?.longTerm ?? {
    async store() {
      return 'mock-id';
    },
    async search() {
      return [];
    },
    async get() {
      return null;
    },
    async delete() {
      return true;
    },
    async forget() {
      return 0;
    },
  };

  const config: AgentConfig = {
    name: 'test-agent',
    model: {
      provider: options.provider,
      model: 'mock-model',
    },
    systemPrompt: options.systemPrompt ?? 'You are a helpful assistant.',
    tools: toolRegistry,
    memory: {
      shortTerm: shortTermMemory,
      longTerm: longTermMemory,
      async consolidate() {},
    },
    maxIterations: options.maxIterations ?? 10,
  };

  return new DefaultAgent(config);
}

// ─── 测试用例 ───────────────────────────────────────

describe('DefaultAgent', () => {
  describe('run()', () => {
    it('应该直接返回 LLM 的回答（无工具调用）', async () => {
      const provider = createMockProvider([
        { content: 'Hello! How can I help you?', finishReason: 'stop' },
      ]);
      const agent = createTestAgent({ provider });

      const result = await agent.run('Hi');

      expect(result.output).toBe('Hello! How can I help you?');
      expect(result.iterations).toBe(1);
      expect(result.stoppedEarly).toBe(false);
      expect(result.messages.length).toBeGreaterThanOrEqual(3); // system + user + assistant
    });

    it('应该支持一次工具调用并返回最终结果', async () => {
      const provider = createMockProvider([
        {
          content: '',
          finishReason: 'tool_calls',
          toolCalls: [
            {
              id: 'call_1',
              name: 'calculator',
              args: { expression: '1+2' },
            },
          ],
        },
        {
          content: 'The result of 1 + 2 is 3.',
          finishReason: 'stop',
        },
      ]);
      const agent = createTestAgent({
        provider,
        tools: [calculatorTool],
      });

      const result = await agent.run('What is 1+2?');

      expect(result.output).toBe('The result of 1 + 2 is 3.');
      expect(result.iterations).toBe(2);
      expect(result.stoppedEarly).toBe(false);
    });

    it('应该支持多轮工具调用', async () => {
      const provider = createMockProvider([
        {
          content: '',
          finishReason: 'tool_calls',
          toolCalls: [
            {
              id: 'call_1',
              name: 'calculator',
              args: { expression: '3*4' },
            },
          ],
        },
        {
          content: '',
          finishReason: 'tool_calls',
          toolCalls: [
            {
              id: 'call_2',
              name: 'calculator',
              args: { expression: '12+5' },
            },
          ],
        },
        {
          content: 'The final result is 17.',
          finishReason: 'stop',
        },
      ]);
      const agent = createTestAgent({
        provider,
        tools: [calculatorTool],
      });

      const result = await agent.run('Calculate 3*4 and then add 5');

      expect(result.output).toBe('The final result is 17.');
      expect(result.iterations).toBe(3);
      expect(result.stoppedEarly).toBe(false);
    });

    it('达到 maxIterations 时应该提前终止', async () => {
      // 一直返回 tool_calls，导致死循环
      const toolCallResponse: LLMResponse = {
        content: '',
        finishReason: 'tool_calls',
        toolCalls: [
          {
            id: 'call_loop',
            name: 'calculator',
            args: { expression: '1+1' },
          },
        ],
      };
      // 提供足够的响应，让循环走到 maxIterations
      const responses = Array(15).fill(toolCallResponse);
      const provider = createMockProvider(responses);
      const agent = createTestAgent({
        provider,
        tools: [calculatorTool],
        maxIterations: 5,
      });

      const result = await agent.run('Keep calculating');

      expect(result.stoppedEarly).toBe(true);
      expect(result.iterations).toBe(5);
    });

    it('工具不存在时应该优雅地报告错误而不是崩溃', async () => {
      const provider = createMockProvider([
        {
          content: '',
          finishReason: 'tool_calls',
          toolCalls: [
            {
              id: 'call_bad',
              name: 'nonexistent_tool',
              args: {},
            },
          ],
        },
        {
          content: 'I see that the tool was not found.',
          finishReason: 'stop',
        },
      ]);
      const agent = createTestAgent({
        provider,
        // 不注册任何工具
      });

      const result = await agent.run('Use a tool');

      // Agent 应该继续执行，不会崩溃
      expect(result.output).toBe('I see that the tool was not found.');
      expect(result.iterations).toBe(2);
    });

    it('工具抛出异常时不应该影响 Agent 运行', async () => {
      const failingTool: Tool = {
        metadata: {
          name: 'failing_tool',
          description: 'A tool that always fails',
          parameters: z.object({}),
        },
        async execute(): Promise<ToolResult> {
          throw new Error('Something went wrong');
        },
      };

      const provider = createMockProvider([
        {
          content: '',
          finishReason: 'tool_calls',
          toolCalls: [
            {
              id: 'call_fail',
              name: 'failing_tool',
              args: {},
            },
          ],
        },
        {
          content: 'The tool failed but I am still running.',
          finishReason: 'stop',
        },
      ]);
      const agent = createTestAgent({
        provider,
        tools: [failingTool],
      });

      const result = await agent.run('Run the failing tool');

      expect(result.output).toBe('The tool failed but I am still running.');
      expect(result.iterations).toBe(2);
    });

    it('应该将对话保存到短期记忆中，供后续 run 使用', async () => {
      const memory = new SlidingWindowMemory(4096);
      const memSystem: MemorySystem = {
        shortTerm: memory,
        longTerm: {
          async store() { return 'mock-id'; },
          async search() { return []; },
          async get() { return null; },
          async delete() { return true; },
          async forget() { return 0; },
        },
        async consolidate() {},
      };

      // 第一次 run：简单的问答
      const provider1 = createMockProvider([
        { content: 'My name is Alice.', finishReason: 'stop' },
      ]);
      const agent1 = createTestAgent({ provider: provider1, memory: memSystem });
      await agent1.run('What is your name?');

      // 第二次 run：用同一个记忆系统，看是否能取到历史
      const provider2 = createMockProvider([
        { content: 'You asked me before.', finishReason: 'stop' },
      ]);
      const agent2 = createTestAgent({ provider: provider2, memory: memSystem });
      const result2 = await agent2.run('Do you remember me?');

      // 记忆里应该有第一次的对话
      expect(memory.getMessages().length).toBeGreaterThan(0);
      expect(result2.iterations).toBe(1);
    });

    it('应该正确触发所有 Hooks', async () => {
      const provider = createMockProvider([
        {
          content: '',
          finishReason: 'tool_calls',
          toolCalls: [
            { id: 'call_hook', name: 'calculator', args: { expression: '2+2' } },
          ],
        },
        { content: 'The answer is 4.', finishReason: 'stop' },
      ]);

      const onBeforeLLMCall = vi.fn();
      const onAfterLLMCall = vi.fn();
      const onBeforeToolCall = vi.fn();
      const onAfterToolCall = vi.fn();
      const onIterationComplete = vi.fn();

      const agent = createTestAgent({ provider, tools: [calculatorTool] });
      agent.hooks = {
        onBeforeLLMCall,
        onAfterLLMCall,
        onBeforeToolCall,
        onAfterToolCall,
        onIterationComplete,
      };

      await agent.run('Calculate 2+2');

      // 2 轮迭代 → LLM 被调用 2 次
      expect(onBeforeLLMCall).toHaveBeenCalledTimes(2);
      expect(onAfterLLMCall).toHaveBeenCalledTimes(2);

      // 1 次工具调用
      expect(onBeforeToolCall).toHaveBeenCalledTimes(1);
      expect(onBeforeToolCall).toHaveBeenCalledWith('calculator', { expression: '2+2' });
      expect(onAfterToolCall).toHaveBeenCalledTimes(1);

      // 2 轮迭代
      expect(onIterationComplete).toHaveBeenCalledTimes(2);
    });

    it('reset() 应该清除短期记忆', async () => {
      const memory = new SlidingWindowMemory(4096);
      const memSystem: MemorySystem = {
        shortTerm: memory,
        longTerm: {
          async store() { return 'mock-id'; },
          async search() { return []; },
          async get() { return null; },
          async delete() { return true; },
          async forget() { return 0; },
        },
        async consolidate() {},
      };

      const provider = createMockProvider([
        { content: 'Hello!', finishReason: 'stop' },
      ]);
      const agent = createTestAgent({ provider, memory: memSystem });

      await agent.run('Hi');
      expect(memory.getMessages().length).toBeGreaterThan(0);

      agent.reset();
      expect(memory.getMessages().length).toBe(0);
    });
  });
});
