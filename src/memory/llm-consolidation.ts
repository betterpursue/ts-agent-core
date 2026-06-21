/**
 * LLM 驱动的记忆合并（LLM-Driven Memory Merging）
 *
 * 用 LLM 替代规则引擎来做两件事：
 * 1. 从对话中提取结构化事实（比规则更精准，能理解隐含信息）
 * 2. 检测新事实与已有记忆之间的关系，自动合并
 *
 * 设计原则：
 * - LLMProvider 可替换（OpenAI / Anthropic / Ollama / mock）
 * - 提取和合并是两个独立阶段，便于测试和调试
 * - 合并策略涵盖四种关系：new / duplicate / merge / contradiction
 * - 置信度机制：LLM 返回 confidence，低于阈值的不存储
 *
 * 使用方式：
 * ```typescript
 * const strategy = new LLMConsolidationStrategy({
 *   provider: openaiProvider,
 *   confidenceThreshold: 0.6,
 * });
 * const facts = await strategy.extract(messages);
 * const merged = await strategy.merge(facts, existingMemories);
 * ```
 */

import type { Message } from '../core/message.js';
import type { LongTermMemoryItem } from '../core/memory.js';
import type { ExtractedFact, ConsolidationStrategy } from './consolidation.js';

// ─── LLM Provider 接口（记忆合并专用） ─────────────────────────

/**
 * 极简 LLM 抽象 —— 只需一个 complete() 方法
 *
 * 与 core/llm.ts 的 LLMProvider 不同，这里只做文本补全，
 * 无需支持 tool calling、streaming 等能力，接口更轻量。
 * 不绑定任何具体模型或 SDK，方便测试和替换。
 */
export interface ConsolidationLLM {
  /** 模型名称（仅用于日志和调试） */
  readonly name: string;

  /**
   * 完成一次 LLM 调用
   *
   * @param system  系统提示词（角色设定）
   * @param prompt  用户提示词（具体任务）
   * @returns       模型回复文本
   */
  complete(system: string, prompt: string): Promise<string>;
}

// ─── OpenAI 实现 ────────────────────────────────────────────────

/**
 * 基于 OpenAI Chat Completions API 的 ConsolidationLLM 实现
 *
 * 使用环境变量 OPENAI_API_KEY 作为 API Key。
 * 可通过 baseURL 配置兼容的代理（如 Ollama、vLLM）。
 */
export interface OpenAIConsolidationConfig {
  apiKey?: string;
  baseURL?: string;
  model?: string;
}

export class OpenAIConsolidation implements ConsolidationLLM {
  readonly name: string;
  private apiKey: string;
  private baseURL: string;
  private model: string;

  constructor(config?: OpenAIConsolidationConfig) {
    this.apiKey = config?.apiKey ?? process.env.OPENAI_API_KEY ?? '';
    this.baseURL = config?.baseURL ?? 'https://api.openai.com/v1';
    this.model = config?.model ?? 'gpt-4o-mini';
    this.name = `openai:${this.model}`;
  }

  async complete(system: string, prompt: string): Promise<string> {
    const url = `${this.baseURL}/chat/completions`;
    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${this.apiKey}`,
      },
      body: JSON.stringify({
        model: this.model,
        messages: [
          { role: 'system', content: system },
          { role: 'user', content: prompt },
        ],
        temperature: 0.3,
      }),
    });

    if (!response.ok) {
      const text = await response.text();
      throw new Error(
        `OpenAI API error (${response.status}): ${text}`
      );
    }

    const data = (await response.json()) as {
      choices: Array<{ message: { content: string } }>;
    };

    return data.choices[0]?.message?.content ?? '';
  }
}

// ─── Mock（测试用） ─────────────────────────────────────────────

/**
 * Mock ConsolidationLLM —— 返回预设的 JSON 响应
 *
 * 用于测试 LLMConsolidationStrategy 的行为，不依赖外部 API。
 */
export class MockConsolidationLLM implements ConsolidationLLM {
  readonly name = 'mock';
  private responses: string[];
  private callIndex = 0;

  constructor(responses: string[]) {
    this.responses = responses;
  }

  async complete(_system: string, _prompt: string): Promise<string> {
    const response = this.responses[this.callIndex % this.responses.length];
    this.callIndex++;
    return response;
  }

  get callCount(): number {
    return this.callIndex;
  }

  reset(): void {
    this.callIndex = 0;
  }
}

// ─── LLM Consolidation 配置 ────────────────────────────────────

export interface LLMConsolidationConfig {
  /** 记忆合并专用的 LLM 抽象 */
  provider: ConsolidationLLM;

  /** 事实置信度阈值（0-1），低于此值的不存储。默认 0.5 */
  confidenceThreshold?: number;

  /** 单次提取的最大事实数。默认 15 */
  maxFactsPerExtraction?: number;

  /** 提取时的批处理大小。默认 10 */
  extractionBatchSize?: number;

  /** 合并时检索的已有记忆数量。默认 20 */
  mergeSearchSize?: number;
}

// ─── LLM 提取的原始事实格式 ────────────────────────────────────

/**
 * LLM 返回的原始事实（JSON parse 前的中间结构）
 */
interface LLMExtractedFact {
  content: string;
  type: 'fact' | 'preference' | 'decision';
  tags: string[];
  confidence: number;
}

/**
 * LLM 合并决策结果
 */
interface LLMMergeDecision {
  factIndex: number;
  action: 'keep' | 'skip' | 'merge_into';
  mergeTargetIndex?: number;
  mergedContent?: string;
  reason: string;
}

// ─── LLM Consolidation Strategy ─────────────────────────────────

/**
 * LLM 驱动的记忆合并策略
 *
 * 相比 SimpleConsolidationStrategy 的核心改进：
 *
 * 1. 语义级提取，而非关键词匹配
 *    规则引擎只能抓取 "我喜欢 X" 这种模板，LLM 能理解 "X 对我来说特别重要"
 *    这样的隐含偏好。
 *
 * 2. 合并阶段处理四种关系
 *    - duplicate：完全重复，跳过
 *    - subsumes：新事实是已有事实的特例，合并为更精确的表述
 *    - generalizes：新事实是已有事实的泛化，保留更全面的那条
 *    - contradiction：矛盾，标记待人工审查
 *    - unrelated：无关，各自保留
 *
 * 3. 置信度过滤
 *    LLM 对每一条事实给出 confidence（0-1），低于阈值的不存储。
 *    这避免了过度存储模糊信息。
 */
export class LLMConsolidationStrategy implements ConsolidationStrategy {
  private provider: ConsolidationLLM;
  private confidenceThreshold: number;
  private maxFactsPerExtraction: number;
  private extractionBatchSize: number;
  private mergeSearchSize: number;

  constructor(config: LLMConsolidationConfig) {
    this.provider = config.provider;
    this.confidenceThreshold = config.confidenceThreshold ?? 0.5;
    this.maxFactsPerExtraction = config.maxFactsPerExtraction ?? 15;
    this.extractionBatchSize = config.extractionBatchSize ?? 10;
    this.mergeSearchSize = config.mergeSearchSize ?? 20;
  }

  // ════════════════════════════════════════════════════════════════
  // Phase 1: 提取
  // ════════════════════════════════════════════════════════════════

  /**
   * 用 LLM 从对话消息中提取事实
   *
   * 流程：
   * 1. 将消息分批（避免超长上下文）
   * 2. 对每批调用 LLM 提取事实
   * 3. 合并结果，按置信度过滤
   * 4. 将 LLM 原始回复映射到项目的 ExtractedFact 结构
   */
  async extract(messages: Message[]): Promise<ExtractedFact[]> {
    if (messages.length === 0) return [];

    const batches = this.batchMessages(messages, this.extractionBatchSize);
    const allFacts: LLMExtractedFact[] = [];

    for (const batch of batches) {
      try {
        const raw = await this.callExtractionLLM(batch);
        const parsed = this.parseExtractionResponse(raw);
        allFacts.push(...parsed);
      } catch (err) {
        console.warn(
          '[LLMConsolidationStrategy] Batch extraction failed, skipping:',
          err instanceof Error ? err.message : String(err)
        );
      }
    }

    // 置信度过滤 + 映射到 ExtractedFact
    return allFacts
      .filter((f) => f.confidence >= this.confidenceThreshold)
      .slice(0, this.maxFactsPerExtraction)
      .map((f) => this.toExtractedFact(f));
  }

  // ════════════════════════════════════════════════════════════════
  // Phase 2: 合并
  // ════════════════════════════════════════════════════════════════

  /**
   * 将新提取的事实与已有长期记忆合并
   *
   * LLM 逐条分析新事实，与相关已有记忆对比，决定：
   * - keep：新信息，直接保留
   * - skip：重复，跳过
   * - merge_into：与某条已有记忆合并为一条更精确的描述
   *
   * 注意：此方法不会修改已有记忆（那是 DefaultMemorySystem 的职责），
   * 只返回"最终要存储的事实列表"。
   */
  async merge(
    newFacts: ExtractedFact[],
    existingMemories: LongTermMemoryItem[],
  ): Promise<ExtractedFact[]> {
    if (newFacts.length === 0) return [];
    if (existingMemories.length === 0) return newFacts;

    // 用 LLM 分析新事实与已有记忆的关系
    try {
      const decisions = await this.callMergeLLM(newFacts, existingMemories);

      // 根据决策构建最终事实列表
      const result: ExtractedFact[] = [];
      const mergedMap = new Map<number, string>();

      for (const d of decisions) {
        if (d.action === 'skip') continue;

        if (d.action === 'merge_into') {
          // 合并后的内容存到事实索引（如果没有指定目标索引，用 factIndex）
          const targetIdx = d.mergeTargetIndex ?? d.factIndex;
          if (d.mergedContent) {
            mergedMap.set(targetIdx, d.mergedContent);
          }
          continue;
        }

        if (d.action === 'keep') {
          const fact = newFacts[d.factIndex];
          if (fact) result.push(fact);
        }
      }

      // 把 merge_into 的结果追加到最终列表
      for (const [idx, content] of mergedMap) {
        const original = newFacts[idx];
        if (original) {
          result.push({
            ...original,
            content,
            importance: Math.min(original.importance + 0.1, 1.0),
          });
        }
      }

      return result;
    } catch (err) {
      console.warn(
        '[LLMConsolidationStrategy] Merge analysis failed, keeping all new facts:',
        err instanceof Error ? err.message : String(err)
      );
      return newFacts;
    }
  }

  // ════════════════════════════════════════════════════════════════
  // LLM 调用
  // ════════════════════════════════════════════════════════════════

  /**
   * 提取阶段 —— LLM 调用
   *
   * 提示词设计思路：
   * - 明确告诉 LLM 它在做什么（memory analyst）
   * - 给出具体的输出格式（JSON）
   * - 说明哪些内容不要提取（问候、确认、闲聊）
   * - 要求给出 confidence 分数
   */
  private async callExtractionLLM(messages: Message[]): Promise<string> {
    const systemPrompt = [
      'You are a memory analyst for an AI agent. Your job is to extract important',
      'information about the user from conversation messages.',
      '',
      'Rules:',
      '- Only extract facts that are clearly stated or strongly implied.',
      '- Skip greetings, confirmations, chit-chat, and obvious pleasantries.',
      '- If a fact is repeated, only extract it once (higher confidence).',
      '- Be precise: extract the exact meaning, don\'t generalize.',
      '',
      'For each fact, provide:',
      '- content: the fact statement (in the user\'s original language)',
      '- type: "fact" | "preference" | "decision"',
      '- tags: 2-5 relevant keywords',
      '- confidence: 0.0 to 1.0 (how certain you are about this fact)',
      '',
      'Return ONLY a valid JSON array. No markdown, no explanation.',
    ].join('\n');

    const messageText = messages
      .map((m) => `[${m.role}]: ${m.content}`)
      .join('\n');

    const prompt = [
      'Extract important facts from these conversation messages:',
      '',
      messageText,
    ].join('\n');

    return this.provider.complete(systemPrompt, prompt);
  }

  /**
   * 合并阶段 —— LLM 调用
   *
   * 把新事实和已有记忆分别编号，让 LLM 逐条分析关系。
   */
  private async callMergeLLM(
    newFacts: ExtractedFact[],
    existingMemories: LongTermMemoryItem[],
  ): Promise<LLMMergeDecision[]> {
    // 选择相关的已有记忆（取最新的 N 条做采样）
    const relevantExisting = existingMemories
      .sort((a, b) => b.metadata.timestamp - a.metadata.timestamp)
      .slice(0, this.mergeSearchSize);

    const systemPrompt = [
      'You are a memory deduplication and merging system.',
      'Analyze NEW facts against EXISTING memories and decide what to do.',
      '',
      'For each new fact (by index), decide:',
      '- "keep": It\'s genuinely new information, store it as-is.',
      '- "skip": It\'s a duplicate or already covered by existing memories.',
      '- "merge_into": It\'s related to an existing memory and should be merged',
      '  into a more precise combined statement.',
      '',
      'Output format (JSON array):',
      '[',
      '  {',
      '    "factIndex": 0,',
      '    "action": "keep" | "skip" | "merge_into",',
      '    "mergeTargetIndex": null,  // set if action is "merge_into"',
      '    "mergedContent": null,     // the combined statement if merge_into',
      '    "reason": "Brief explanation"',
      '  }',
      ']',
    ].join('\n');

    const promptNew = newFacts
      .map((f, i) => `[${i}] type=${f.type}, tags=[${f.tags.join(', ')}]\n  "${f.content}"`)
      .join('\n\n');

    const promptExisting = relevantExisting
      .map((m, i) => `[E${i}] "${m.content}" (type: ${m.metadata.type})`)
      .join('\n');

    const prompt = [
      'NEW FACTS to analyze:',
      '',
      promptNew,
      '',
      '---',
      '',
      'EXISTING MEMORIES to compare against:',
      '',
      promptExisting,
      '',
      '---',
      'Return your decisions as a JSON array.',
    ].join('\n');

    const raw = await this.provider.complete(systemPrompt, prompt);
    return this.parseMergeResponse(raw, newFacts.length);
  }

  // ════════════════════════════════════════════════════════════════
  // 解析
  // ════════════════════════════════════════════════════════════════

  /**
   * 解析 LLM 返回的提取 JSON
   *
   * 带防崩溃处理：如果 JSON 解析失败，尝试从中提取 JSON 数组部分。
   */
  private parseExtractionResponse(raw: string): LLMExtractedFact[] {
    const json = this.extractJSON(raw);
    if (!json) {
      console.warn(
        '[LLMConsolidationStrategy] Failed to parse extraction response:',
        raw.slice(0, 200)
      );
      return [];
    }

    try {
      const parsed = JSON.parse(json);

      if (!Array.isArray(parsed)) {
        console.warn('[LLMConsolidationStrategy] Extraction response is not an array');
        return [];
      }

      return parsed.map((f: Record<string, unknown>) => ({
        content: String(f.content ?? ''),
        type: this.normalizeType(f.type),
        tags: Array.isArray(f.tags) ? f.tags.map(String) : [],
        confidence: Number(f.confidence) ?? 0.5,
      }));
    } catch {
      console.warn('[LLMConsolidationStrategy] JSON parse error in extraction response');
      return [];
    }
  }

  /**
   * 解析 LLM 返回的合并决策 JSON
   */
  private parseMergeResponse(
    raw: string,
    expectedCount: number,
  ): LLMMergeDecision[] {
    const json = this.extractJSON(raw);
    if (!json) {
      throw new Error('Failed to extract JSON from merge response');
    }

    try {
      const parsed = JSON.parse(json);

      if (!Array.isArray(parsed)) {
        throw new Error('Merge response is not an array');
      }

      return parsed
        .filter(
          (d: Record<string, unknown>) =>
            typeof d.factIndex === 'number' &&
            d.factIndex >= 0 &&
            d.factIndex < expectedCount &&
            ['keep', 'skip', 'merge_into'].includes(String(d.action))
        )
        .map((d: Record<string, unknown>) => ({
          factIndex: Number(d.factIndex),
          action: d.action as 'keep' | 'skip' | 'merge_into',
          mergeTargetIndex:
            d.mergeTargetIndex !== null && d.mergeTargetIndex !== undefined
              ? Number(d.mergeTargetIndex)
              : undefined,
          mergedContent:
            d.mergedContent !== null && d.mergedContent !== undefined
              ? String(d.mergedContent)
              : undefined,
          reason: String(d.reason ?? ''),
        }));
    } catch (err) {
      throw new Error(
        `Failed to parse merge response: ${err instanceof Error ? err.message : String(err)}`
      );
    }
  }

  // ════════════════════════════════════════════════════════════════
  // 辅助方法
  // ════════════════════════════════════════════════════════════════

  /**
   * 从 LLM 回复中提取 JSON 数组
   *
   * LLM 有时会在 JSON 外面加 markdown 代码块或多余文字。
   * 这个方法会找到第一个 [ 到最后一个 ] 之间的内容。
   */
  private extractJSON(text: string): string | null {
    // 尝试匹配 ```json ... ``` 代码块
    const codeBlockMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (codeBlockMatch) {
      return codeBlockMatch[1].trim();
    }

    // 尝试直接找到 JSON 数组
    const start = text.indexOf('[');
    const end = text.lastIndexOf(']');
    if (start !== -1 && end !== -1 && end > start) {
      return text.slice(start, end + 1);
    }

    return null;
  }

  /**
   * 规范化 LLM 返回的类型字符串
   */
  private normalizeType(type: unknown): 'fact' | 'preference' | 'decision' {
    const t = String(type ?? '').toLowerCase();
    if (t === 'preference') return 'preference';
    if (t === 'decision') return 'decision';
    return 'fact';
  }

  /**
   * 将 LLMExtractedFact 映射到项目的 ExtractedFact
   *
   * importance 从 confidence 映射而来：
   * - confidence 0.5 → importance 0.3
   * - confidence 1.0 → importance 0.8
   * - 线性映射 + 类型加成
   */
  private toExtractedFact(f: LLMExtractedFact): ExtractedFact {
    let importance = 0.3 + f.confidence * 0.5;

    // 类型加成
    if (f.type === 'preference') importance += 0.1;
    if (f.type === 'decision') importance += 0.05;

    return {
      content: f.content,
      type: f.type,
      tags: f.tags,
      importance: Math.min(importance, 1.0),
    };
  }

  /**
   * 将消息列表分批
   */
  private batchMessages(messages: Message[], batchSize: number): Message[][] {
    const batches: Message[][] = [];
    for (let i = 0; i < messages.length; i += batchSize) {
      batches.push(messages.slice(i, i + batchSize));
    }
    return batches;
  }
}
