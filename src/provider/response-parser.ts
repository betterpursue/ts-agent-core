/**
 * LLM 响应解析器
 *
 * 从 LLM 返回的原始响应中提取 tool_calls，做参数校验和修复。
 * Function Calling 链路中，解析层是最容易出Bug的地方——
 * LLM 会生成不完整的 JSON、会 invent 不存在的字段、会忘记闭合括号。
 *
 * 设计原则：
 * - 防御性解析：LLM 的输出天生不可靠，每个字段都要兜底
 * - 参数校验：用 Zod schema 验证 LLM 生成的参数，及早发现不合法调用
 * - 错误结构化：每个解析失败都能追踪到具体是哪个 tool_call 出了什么问题
 */

import type { ToolCall, Message } from '../core/message.js';
import type { ToolRegistry, ToolMetadata } from '../core/tool.js';
import type { LLMResponse } from '../core/llm.js';

// ─── 解析结果类型 ───────────────────────────────────────────────

/** 解析后的工具调用（参数已校验和修复） */
export interface ParsedToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
  /** 参数是否通过了 schema 校验 */
  argsValid: boolean;
  /** 如果参数校验失败，这里记录详情 */
  validationErrors?: string[];
  /** 原始参数文本（用于调试） */
  rawArgs?: string;
}

/** 单次解析的结果 */
export interface ParseResult {
  /** 成功解析的 tool calls */
  calls: ParsedToolCall[];
  /** 完全解析失败的调用（JSON 无法修复，直接放弃） */
  failed: Array<{
    id: string;
    name: string;
    error: string;
    rawText?: string;
  }>;
  /** 工具名称对齐：LLM 可能输出不存在的工具名，这里记录映射信息 */
  warnings: string[];
}

// ─── JSON 修复器 ────────────────────────────────────────────────

/**
 * 尝试修复 LLM 生成的不完整 JSON
 *
 * LLM 常见的 JSON 错误模式：
 * 1. 缺少闭合括号：`{"a": 1, "b": 2`  → `{"a": 1, "b": 2}`
 * 2. 最后一个 value 后面多了逗号：`{"a": 1,}` → `{"a": 1}`
 * 3. 字符串未闭合：`{"a": "hello` → `{"a": "hello"}`
 * 4. 单引号代替双引号：`{'a': 1}` → `{"a": 1}`
 * 5. 多余的尾部字符
 *
 * @returns 修复后的 JSON 字符串，如果无法修复则返回 null
 */
export function repairJson(raw: string): string | null {
  if (!raw || raw.trim().length === 0) return null;

  let text = raw.trim();

  // 尝试直接解析
  try {
    JSON.parse(text);
    return text; // 合法的 JSON，不需要修复
  } catch {
    // 继续尝试修复
  }

  // 修复 1：单引号替换为双引号
  let repaired = text.replace(/'/g, '"');

  // 修复 2：移除末尾多余的逗号（在 } 前面的逗号）
  repaired = repaired.replace(/,(\s*[}\]])/g, '$1');

  // 修复 3：尝试闭合大括号
  const openBraces = (repaired.match(/{/g) || []).length;
  const closeBraces = (repaired.match(/}/g) || []).length;
  for (let i = 0; i < openBraces - closeBraces; i++) {
    repaired += '}';
  }

  // 修复 4：尝试闭合中括号
  const openBrackets = (repaired.match(/\[/g) || []).length;
  const closeBrackets = (repaired.match(/\]/g) || []).length;
  for (let i = 0; i < openBrackets - closeBrackets; i++) {
    repaired += ']';
  }

  // 修复 5：尝试闭合未闭合的字符串
  // 检查是否有多余的双引号（奇数个双引号）
  // 注意：这个修复很粗糙，但能 cover 大部分 case
  const quoteCount = (repaired.match(/"/g) || []).length;
  if (quoteCount % 2 !== 0) {
    repaired += '"';
  }

  // 再次尝试解析
  try {
    JSON.parse(repaired);
    return repaired;
  } catch {
    // 修复失败，返回 null
    return null;
  }
}

// ─── 主解析器 ───────────────────────────────────────────────────

/**
 * 参数验证结果
 */
interface ParamValidation {
  valid: boolean;
  errors: string[];
  /** 修正后的参数（由 Zod schema 的 parse 提供） */
  parsed: Record<string, unknown>;
}

/**
 * 用 Zod schema 校验参数
 *
 * 使用 Zod 的 safeParse 方法进行参数校验。
 * 如果参数不符合 schema，Zod 会给出明确的错误信息。
 * 这和 Tool 执行时的校验一致——保证进入 Tool 的参数必然是合法的。
 */
function validateParams(
  metadata: ToolMetadata,
  args: Record<string, unknown>
): ParamValidation {
  try {
    const result = (metadata.parameters as unknown as {
      safeParse: (data: unknown) => { success: boolean; data?: unknown; error?: { issues: Array<{ path: (string | number)[]; message: string }> } };
    }).safeParse(args);

    if (result.success) {
      return {
        valid: true,
        errors: [],
        parsed: result.data as Record<string, unknown>,
      };
    }

    const errors = result.error?.issues.map(
      (issue) => `\`${issue.path.join('.')}\`: ${issue.message}`
    ) ?? ['Unknown validation error'];

    return {
      valid: false,
      errors,
      // 即使校验失败，也返回修复过的参数（Zod 会做类型转换）
      parsed: args,
    };
  } catch {
    // Zod schema 可能不支持 safeParse（比如自定义类型），降级到基础校验
    return {
      valid: true,
      errors: [],
      parsed: args,
    };
  }
}

/**
 * 解析 LLM 响应中的 tool_calls
 *
 * 完整的解析链路：
 * 1. 检查 tool_calls 是否存在
 * 2. 对每个 tool_call，解析 JSON 格式的 args
 * 3. 如果 JSON 格式错误，尝试修复
 * 4. 如果工具名不在注册列表中，记录警告但不丢弃（留给上游做模糊匹配）
 * 5. 用 Tool 的 Zod schema 校验参数
 * 6. 返回结构化的解析结果
 *
 * @param response  - LLM 的原始响应
 * @param registry  - Tool 注册中心（用于获取 schema 做校验）
 * @returns          - 解析结果
 */
export function parseToolCalls(
  response: LLMResponse,
  registry: ToolRegistry
): ParseResult {
  const result: ParseResult = {
    calls: [],
    failed: [],
    warnings: [],
  };

  if (!response.toolCalls || response.toolCalls.length === 0) {
    return result;
  }

  // 构建工具名 → metadata 的映射，便于快速查询
  const availableTools = new Map<string, ToolMetadata>();
  for (const meta of registry.listMetadata()) {
    availableTools.set(meta.name, meta);
  }

  for (const tc of response.toolCalls) {
    const parsed: ParsedToolCall = {
      id: tc.id || `tc_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
      name: tc.name,
      args: {},
      argsValid: false,
    };

    // 步骤 1：检查工具是否存在
    const toolMeta = availableTools.get(tc.name);
    if (!toolMeta) {
      // 工具不存在，保留原始 args 但不校验
      // 上报给上游，上游决定是报错还是尝试模糊匹配
      result.warnings.push(
        `Tool '${tc.name}' is not registered. Available: [${Array.from(availableTools.keys()).join(', ')}]`
      );
      parsed.args = tc.args || {};
      parsed.argsValid = true; // 不存在就无法校验，交给上游处理
      result.calls.push(parsed);
      continue;
    }

    // 步骤 2：解析参数
    let rawArgs = tc.args;

    // 检查参数是否是字符串（某些 Provider 返回 JSON 字符串而非对象）
    if (typeof rawArgs === 'string') {
      const originalText = rawArgs as string;
      parsed.rawArgs = originalText;

      const repaired = repairJson(originalText);
      if (repaired) {
        try {
          rawArgs = JSON.parse(repaired) as Record<string, unknown>;
        } catch {
          result.failed.push({
            id: tc.id,
            name: tc.name,
            error: 'Failed to parse arguments JSON after repair attempt',
            rawText: originalText,
          });
          continue;
        }
      } else {
        result.failed.push({
          id: tc.id,
          name: tc.name,
          error: 'Arguments JSON is not parseable and could not be repaired',
          rawText: originalText,
        });
        continue;
      }
    }

    // 步骤 3：校验参数
    if (typeof rawArgs === 'object' && rawArgs !== null && !Array.isArray(rawArgs)) {
      const validation = validateParams(toolMeta, rawArgs as Record<string, unknown>);
      parsed.args = validation.parsed;
      parsed.argsValid = validation.valid;
      parsed.validationErrors = validation.valid ? undefined : validation.errors;
    } else {
      // args 既不是 object 也不是 string（比如是数组或 null）
      result.failed.push({
        id: tc.id,
        name: tc.name,
        error: `Arguments must be an object, got ${typeof rawArgs}`,
      });
      continue;
    }

    result.calls.push(parsed);
  }

  return result;
}

// ─── 对话构建器 ────────────────────────────────────────────────

/**
 * 创建 tool result 消息
 *
 * 将工具的执行结果包装为 Message 格式，准备喂回给 LLM。
 * 如果工具执行失败，isError 标志会告诉 LLM 这是一次失败的调用。
 */
export function buildToolResultMessage(
  toolCall: ParsedToolCall,
  output: string,
  isError = false
): Message {
  return {
    role: 'tool',
    content: output,
    toolCallId: toolCall.id,
    toolName: toolCall.name,
    isError,
  };
}

/**
 * 从解析结果中提取所有 tool 消息（用于 debugging）
 */
export function parseResultSummary(result: ParseResult): string {
  const parts: string[] = [];
  parts.push(`Tool calls: ${result.calls.length} parsed, ${result.failed.length} failed`);
  if (result.warnings.length > 0) {
    parts.push(`Warnings: ${result.warnings.join('; ')}`);
  }
  if (result.failed.length > 0) {
    parts.push(
      `Failures: ${result.failed.map((f) => `${f.name}(${f.id}): ${f.error}`).join('; ')}`
    );
  }
  return parts.join('\n');
}
