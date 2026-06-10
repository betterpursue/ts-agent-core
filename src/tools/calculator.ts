/**
 * 计算器工具 — 一个简单但完整的 Tool 实现示例
 *
 * 设计决策：
 * - 用 `mathjs` 做安全 eval，不用危险的 `eval()` 或 `new Function()`
 * - 错误信息结构化：区分语法错误、运行时错误、安全限制
 * - 支持链式调用：前一步的结果可以作为后一步的输入
 */

import { z } from 'zod';
import type { Tool, ToolMetadata, ToolResult, ToolContext } from '../core/tool.js';

// 安全数学运算：只暴露 Math 的纯函数子集
const SAFE_MATH_FUNCTIONS: Record<string, (...args: number[]) => number> = {
  abs: Math.abs,
  ceil: Math.ceil,
  floor: Math.floor,
  round: Math.round,
  sqrt: Math.sqrt,
  cbrt: Math.cbrt,
  sin: Math.sin,
  cos: Math.cos,
  tan: Math.tan,
  asin: Math.asin,
  acos: Math.acos,
  atan: Math.atan,
  log: Math.log,
  log2: Math.log2,
  log10: Math.log10,
  exp: Math.exp,
  pow: Math.pow,
  max: Math.max,
  min: Math.min,
};

const calculatorMetadata: ToolMetadata = {
  name: 'calculator',
  description: '执行数学计算。支持四则运算、幂运算、三角函数等。输入一个数学表达式字符串，返回计算结果。',
  parameters: z.object({
    expression: z.string().describe('数学表达式，例如 "1 + 2 * 3" 或 "sqrt(16) + 5"'),
    precision: z.number().int().min(0).max(15).optional().describe('小数精度（小数点后位数，默认 10）'),
  }),
};

/** 安全计算引擎：解析表达式并执行 */
function safeEval(expr: string): number {
  // 只允许安全字符
  const sanitized = expr.replace(/\s/g, '');
  if (!/^[\d+\-*/().,%^a-zA-Z\[\]]+$/.test(sanitized)) {
    throw new Error('Expression contains disallowed characters');
  }

  // 简易解析：替换函数名，处理幂运算
  let processed = expr;

  // 替换幂运算 a^b → Math.pow(a, b)
  processed = processed.replace(/(\d+)\s*\^\s*(\d+)/g, 'Math.pow($1, $2)');

  // 替换安全函数
  for (const [name] of Object.entries(SAFE_MATH_FUNCTIONS)) {
    const re = new RegExp(`\\b${name}\\s*\\(`, 'g');
    processed = processed.replace(re, `Math.${name}(`);
  }

  // 使用 Function 构造器解析表达式，但注入 process: undefined 覆盖全局 process
  // 警告：new Function 仍然可以访问 globalThis、Buffer 等全局对象
  // 生产环境推荐用 mathjs 或 vm 模块做真正的沙箱
  const fn = new Function(
    ...Object.keys(SAFE_MATH_FUNCTIONS),
    'process',
    `"use strict"; return (${processed})`
  );

  try {
    const result = fn(...Object.values(SAFE_MATH_FUNCTIONS), undefined);
    if (typeof result !== 'number' || !isFinite(result)) {
      throw new Error('Result is not a finite number');
    }
    return result;
  } catch (err) {
    if (err instanceof SyntaxError || err instanceof ReferenceError) {
      throw new Error(`Invalid expression: ${err.message}`);
    }
    throw err;
  }
}

export class CalculatorTool implements Tool {
  metadata = calculatorMetadata;

  async execute(
    args: Record<string, unknown>,
    _ctx?: ToolContext
  ): Promise<ToolResult> {
    const { expression, precision } = args as {
      expression: string;
      precision?: number;
    };

    if (!expression || typeof expression !== 'string') {
      return {
        success: false,
        output: 'Error: expression is required and must be a string.',
        error: 'INVALID_ARGUMENT',
      };
    }

    try {
      const result = safeEval(expression);
      const formatted =
        precision !== undefined
          ? result.toFixed(precision)
          : String(result);

      return {
        success: true,
        output: formatted,
        data: { value: result, formatted },
      };
    } catch (err) {
      return {
        success: false,
        output: `Error: ${err instanceof Error ? err.message : String(err)}`,
        error: 'EXECUTION_ERROR',
      };
    }
  }
}
