/**
 * 消息模块 — Agent 对话的基本单元
 *
 * 设计原则：
 * - 兼容 OpenAI/Anthropic 等多厂商的消息格式
 * - 预留 tool_call / tool_result 两种特殊角色
 * - 与具体的 LLM Provider 解耦（Provider 负责转换成本地格式）
 * - 包含消息校验、Token 估算、窗口管理等运行时工具
 */

// ─── 类型定义 ────────────────────────────────────────────────────

/** 消息角色 */
export type MessageRole = 'system' | 'user' | 'assistant' | 'tool';

/** Tool 调用请求（嵌入在 assistant 消息中） */
export interface ToolCall {
  id: string;
  name: string;
  args: Record<string, unknown>;
}

/** 一条完整消息 */
export interface Message {
  role: MessageRole;
  content: string;
  toolCalls?: ToolCall[];
  toolCallId?: string;
  toolName?: string;
  isError?: boolean;
  metadata?: Record<string, unknown>;
}

/** 对话上下文 —— 消息列表的封装，方便做 token 预算控制 */
export interface ConversationContext {
  messages: Message[];
  /** 预估的 token 总数（由 TokenEstimator 或 Provider 填充） */
  estimatedTokens?: number;
  /** 系统提示词（独立字段，方便后续做 prompt 优化压缩） */
  systemPrompt?: string;
}

// ─── 消息校验 ────────────────────────────────────────────────────

/**
 * 消息校验结果
 */
export interface MessageValidation {
  valid: boolean;
  errors: string[];
}

/**
 * 校验单条消息的结构合法性
 *
 * 检查内容：
 * - 必填字段是否存在
 * - role 和 tool 相关字段的逻辑一致性
 * - toolCalls 的结构完整性
 */
export function validateMessage(msg: Message): MessageValidation {
  const errors: string[] = [];

  if (!msg.role) {
    errors.push('Message must have a role');
    return { valid: false, errors };
  }

  const validRoles: MessageRole[] = ['system', 'user', 'assistant', 'tool'];
  if (!validRoles.includes(msg.role)) {
    errors.push(`Invalid role: ${msg.role}`);
  }

  if (typeof msg.content !== 'string') {
    errors.push('Message content must be a string');
  }

  // role 与 tool 字段的逻辑一致性
  if (msg.role === 'tool') {
    if (!msg.toolCallId) {
      errors.push('Tool message must have toolCallId');
    }
    if (!msg.toolName) {
      errors.push('Tool message must have toolName');
    }
  }

  if (msg.role === 'assistant') {
    if (msg.toolCallId) {
      errors.push('Assistant message should not have toolCallId (tool result only)');
    }
    if (msg.toolName) {
      errors.push('Assistant message should not have toolName (tool result only)');
    }
  }

  // toolCalls 结构校验
  if (msg.toolCalls && msg.toolCalls.length > 0) {
    if (msg.role !== 'assistant') {
      errors.push('Only assistant messages can have toolCalls');
    }
    for (let i = 0; i < msg.toolCalls.length; i++) {
      const tc = msg.toolCalls[i];
      if (!tc.id) errors.push(`toolCalls[${i}].id is required`);
      if (!tc.name) errors.push(`toolCalls[${i}].name is required`);
      if (tc.args === undefined || tc.args === null) {
        errors.push(`toolCalls[${i}].args is required`);
      }
    }
  }

  return { valid: errors.length === 0, errors };
}

/**
 * 校验整个消息序列的合法性
 *
 * 验证规则（符合 LLM Chat Completions 格式规范）：
 * - 第一条消息若不是 system，则不能有 tool 相关引用
 * - 不能连续出现两个相同 non-tool 角色
 * - tool 消息前必须是 assistant（带 toolCalls）
 * - 消息序列不能为空
 */
export function validateConversation(messages: Message[]): MessageValidation {
  const errors: string[] = [];

  if (messages.length === 0) {
    errors.push('Conversation must have at least one message');
    return { valid: false, errors };
  }

  // 单条消息校验
  for (let i = 0; i < messages.length; i++) {
    const result = validateMessage(messages[i]);
    errors.push(...result.errors.map((e) => `messages[${i}]: ${e}`));
  }

  // 序列约束
  for (let i = 1; i < messages.length; i++) {
    const prev = messages[i - 1];
    const curr = messages[i];

    // tool 消息必须有对应的 assistant tool_calls
    if (curr.role === 'tool') {
      const hasMatchingToolCall = prev.role === 'assistant' &&
        prev.toolCalls?.some((tc) => tc.id === curr.toolCallId);
      if (!hasMatchingToolCall) {
        errors.push(
          `messages[${i}]: tool message '${curr.toolName}' has no matching ` +
          `tool_call in messages[${i - 1}]`
        );
      }
    }

    // 不能连续非 tool 消息角色相同
    const SAME_ROLE_BLOCKS: Set<string> = new Set(['user', 'assistant']);
    if (
      SAME_ROLE_BLOCKS.has(prev.role) &&
      prev.role === curr.role &&
      !prev.toolCalls
    ) {
      errors.push(
        `messages[${i - 1}] and messages[${i}]: consecutive '${prev.role}' messages ` +
        `without tool calls`
      );
    }
  }

  return { valid: errors.length === 0, errors };
}

// ─── Token 估算 ──────────────────────────────────────────────────

/**
 * Token 估算器
 *
 * 提供粗略的 token 计数，用于消息窗口的预算管理。
 * 精确计数应由 LLM Provider 的 tokenizer 完成，
 * 这里的估算用于在调用 Provider 之前做预判。
 */
export class TokenEstimator {
  /** 中文字符权重（中文比英文消耗更多 token） */
  private chineseCharWeight: number;
  /** 每条消息的固定 overhead（角色标识、元数据等） */
  private messageOverhead: number;

  constructor(options?: {
    chineseCharWeight?: number;
    messageOverhead?: number;
  }) {
    this.chineseCharWeight = options?.chineseCharWeight ?? 2.0;
    this.messageOverhead = options?.messageOverhead ?? 4;
  }

  /**
   * 估算一段文本的 token 数
   *
   * 算法：
   * - 中文字符（Unicode 范围 \u4e00-\u9fff）：每个 ≈ chineseCharWeight tokens
   * - 其他字符（英文字母、数字、标点、空格）：每 4 个 ≈ 1 token
   * - 空白字符不计数
   */
  estimateText(text: string): number {
    if (!text) return 0;
    let tokens = 0;
    for (const char of text) {
      if (/[\u4e00-\u9fff]/.test(char)) {
        tokens += this.chineseCharWeight;
      } else if (!/\s/.test(char)) {
        tokens += 0.25; // 每 4 个非中文非空白字符 ≈ 1 token
      }
    }
    return Math.ceil(tokens);
  }

  /**
   * 估算一条消息的 token 数
   */
  estimateMessage(msg: Message): number {
    let tokens = this.estimateText(msg.content) + this.messageOverhead;

    if (msg.toolCalls) {
      for (const tc of msg.toolCalls) {
        tokens += this.estimateText(tc.name);
        tokens += this.estimateText(JSON.stringify(tc.args));
        tokens += 2; // tool_call id overhead
      }
    }

    if (msg.role === 'tool') {
      tokens += this.estimateText(msg.toolName ?? '');
    }

    return tokens;
  }

  /**
   * 估算一组消息的总 token 数
   */
  estimateMessages(messages: Message[]): number {
    return messages.reduce((sum, m) => sum + this.estimateMessage(m), 0);
  }

  /**
   * 更新 ConversationContext 的 estimatedTokens
   */
  estimateContext(ctx: ConversationContext): ConversationContext {
    return {
      ...ctx,
      estimatedTokens: this.estimateMessages(ctx.messages),
    };
  }
}

/** 默认的 token 估算器实例 */
export const defaultTokenEstimator = new TokenEstimator();

// ─── 消息窗口管理 ────────────────────────────────────────────────

/**
 * 消息修剪策略
 */
export type TrimStrategy = 'drop_oldest' | 'drop_oldest_pair' | 'summarize';

/**
 * 消息窗口管理器
 *
 * 管理一个消息列表，在不超过 token 预算的前提下，
 * 智能地决定保留哪些消息、裁剪哪些消息。
 *
 * 相比简单的 SlidingWindowMemory，它更聪明：
 * - 始终保留 system prompt
 * - 完整保留最近的 N 轮对话
 * - 裁剪中间轮次时，保留 tool 调用和结果的完整性
 */
export class MessageWindow {
  private messages: Message[];
  private maxTokens: number;
  private estimator: TokenEstimator;
  private trimStrategy: TrimStrategy;

  constructor(options?: {
    maxTokens?: number;
    estimator?: TokenEstimator;
    trimStrategy?: TrimStrategy;
  }) {
    this.messages = [];
    this.maxTokens = options?.maxTokens ?? 4096;
    this.estimator = options?.estimator ?? defaultTokenEstimator;
    this.trimStrategy = options?.trimStrategy ?? 'drop_oldest_pair';
  }

  /** 添加消息 */
  add(msg: Message): void {
    this.messages.push(msg);
    this.trim();
  }

  /** 批量添加 */
  addMany(msgs: Message[]): void {
    this.messages.push(...msgs);
    this.trim();
  }

  /** 获取当前所有消息 */
  getAll(): Message[] {
    return [...this.messages];
  }

  /** 清空 */
  clear(): void {
    this.messages = [];
  }

  /** 当前 token 数 */
  get tokenCount(): number {
    return this.estimator.estimateMessages(this.messages);
  }

  /** 是否为空 */
  get isEmpty(): boolean {
    return this.messages.length === 0;
  }

  /** 消息数量 */
  get length(): number {
    return this.messages.length;
  }

  /**
   * 执行修剪，直到 token 数不超过预算
   */
  private trim(): void {
    while (
      this.estimator.estimateMessages(this.messages) > this.maxTokens &&
      this.messages.length > 1
    ) {
      switch (this.trimStrategy) {
        case 'drop_oldest':
          this.dropOldest();
          break;
        case 'drop_oldest_pair':
          this.dropOldestPair();
          break;
        case 'summarize':
          // 摘要策略需要外部 summarize 函数，暂退化为 drop_oldest_pair
          this.dropOldestPair();
          break;
      }
    }
  }

  /**
   * 移除最早的非 system 消息
   *
   * 如果移除的是 assistant，同时移除它对应的 tool 结果。
   * 如果移除的是 tool，只移除它自己（保留 assistant tool_calls）。
   */
  private dropOldest(): void {
    const nonSystemIdx = this.messages.findIndex((m) => m.role !== 'system');
    if (nonSystemIdx === -1) return;

    const removed = this.messages[nonSystemIdx];
    this.messages.splice(nonSystemIdx, 1);

    // 如果移除的是 assistant 且有 toolCalls，也要移除对应的 tool 结果
    if (removed.role === 'assistant' && removed.toolCalls) {
      const toolCallIds = new Set(removed.toolCalls.map((tc) => tc.id));
      this.messages = this.messages.filter(
        (m) => !(m.role === 'tool' && m.toolCallId && toolCallIds.has(m.toolCallId))
      );
    }
  }

  /**
   * 移除最早的完整对话轮次（user + assistant 对）
   *
   * 在 agent 对话中，最自然的裁剪粒度是"一轮对话"：
   * user 提问 → assistant 回答（可能有 tool calls）/ tool 结果。
   * 这样裁剪后，对话仍然保持完整，不会出现 user 提问后没有回答的情况。
   */
  private dropOldestPair(): void {
    let startIdx = 0;

    // 跳过开头的 system messages
    while (
      startIdx < this.messages.length &&
      this.messages[startIdx].role === 'system'
    ) {
      startIdx++;
    }

    if (startIdx >= this.messages.length) return;

    // 找到最早的一轮：user → (assistant + tools)*
    const userIdx = this.messages.findIndex(
      (m, i) => i >= startIdx && m.role === 'user'
    );
    if (userIdx === -1) {
      // 没有 user message，只能删最早的一条
      this.messages.splice(startIdx, 1);
      return;
    }

    // 从 userIdx 开始，找到这一轮的结束
    let endIdx = userIdx + 1;
    while (endIdx < this.messages.length) {
      const role = this.messages[endIdx].role;
      if (role === 'user' || role === 'system') break; // 新的一轮开始
      endIdx++;
    }

    this.messages.splice(userIdx, endIdx - userIdx);
  }
}

// ─── 工厂函数 ────────────────────────────────────────────────────

export function systemMessage(content: string): Message {
  return { role: 'system', content };
}

export function userMessage(content: string): Message {
  return { role: 'user', content };
}

export function assistantMessage(
  content: string,
  toolCalls?: ToolCall[]
): Message {
  return { role: 'assistant', content, toolCalls };
}

export function toolResultMessage(
  toolCallId: string,
  toolName: string,
  content: string,
  isError?: boolean
): Message {
  return { role: 'tool', content, toolCallId, toolName, isError };
}
