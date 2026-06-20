/**
 * Embedding Provider —— 向量的"翻译官"
 *
 * Embedding 的本质是把文本映射到高维空间的向量坐标。
 * 语义相近的文本，在空间中距离也近——这正是语义检索的基石。
 *
 * 设计原则：
 * - EmbeddingProvider 接口可替换，不绑定任何厂商
 * - OpenAIEmbeddingProvider 用原生 fetch，零外部依赖
 * - 支持 batch embedding（embedMany）减少 API 调用
 * - 内置重试和错误处理，复用 OpenAIProvider 的异常体系
 *
 * 为什么用原生 fetch 而非 OpenAI SDK：
 * - 保持整个项目的零依赖约束一致
 * - embedding API 只有一个 endpoint，不需要 SDK
 * - 减少 node_modules 体积和版本冲突风险
 */

import type { EmbeddingProvider } from '../core/memory.js';

// ─── 配置 ───────────────────────────────────────────────────────

export interface OpenAIEmbeddingProviderConfig {
  /** OpenAI API Key（默认从 OPENAI_API_KEY 环境变量读取） */
  apiKey?: string;
  /** API 基础 URL（默认 https://api.openai.com/v1） */
  baseUrl?: string;
  /** 请求超时 ms（默认 30000） */
  timeout?: number;
  /** 最大重试次数（默认 2） */
  maxRetries?: number;
  /** 模型 ID（默认 text-embedding-3-small） */
  model?: string;
  /** 向量维度（默认 384，text-embedding-3-small 支持截断） */
  dimensions?: number;
}

// ─── 内部类型 ───────────────────────────────────────────────────

interface OpenAIEmbeddingResponse {
  object: 'list';
  data: Array<{
    object: 'embedding';
    index: number;
    embedding: number[];
  }>;
  model: string;
  usage: {
    prompt_tokens: number;
    total_tokens: number;
  };
}

// ─── 错误 ───────────────────────────────────────────────────────

export class EmbeddingError extends Error {
  constructor(
    message: string,
    public readonly statusCode?: number
  ) {
    super(message);
    this.name = 'EmbeddingError';
  }
}

// ─── 默认常量 ───────────────────────────────────────────────────

export const EMBEDDING_DEFAULTS = {
  model: 'text-embedding-3-small',
  dimensions: 384,
  timeout: 30_000,
  maxRetries: 2,
} as const;

// ─── 工具函数 ───────────────────────────────────────────────────

/**
 * 余弦相似度 —— 衡量两个向量在方向上的"靠近程度"
 *
 * 公式：cos(θ) = (A · B) / (|A| × |B|)
 *
 * 为什么是余弦而非欧氏距离：
 * - 余弦看方向不看幅度，embedding 的绝对值通常没有语义意义
 * - 归一化后取值范围为 [-1, 1]，1 表示完全同向，-1 表示完全反向
 * - 实际检索中很少出现负值（大部分文本互不矛盾），但保留它更通用
 *
 * 物理直觉：
 * 把向量想象成从原点射出的箭头。余弦相似度衡量的是箭头之间的夹角。
 * 夹角越小，方向越一致，语义越接近。
 * 长度（模长）被忽略——因为句子"我喜欢猫"和"我超级无敌喜欢猫"
 * 在语义上接近，只是强度不同。
 */
export function cosineSimilarity(a: number[], b: number[]): number {
  if (a.length !== b.length) {
    throw new EmbeddingError(
      `Dimension mismatch: ${a.length} vs ${b.length}`
    );
  }

  let dotProduct = 0;
  let normA = 0;
  let normB = 0;

  // 一次遍历算完三个累加值
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  const denominator = Math.sqrt(normA) * Math.sqrt(normB);

  // 避免除零：零向量与任何向量的相似度为 0
  if (denominator === 0) return 0;

  // 余弦值计算有浮点误差，卡到 [-1, 1] 区间
  const raw = dotProduct / denominator;
  return Math.max(-1, Math.min(1, raw));
}

/**
 * L2 归一化：将向量缩放到单位长度
 *
 * 将向量每个分量除以模长，使得 |v'| = 1。
 * 归一化后的向量做点积等价于余弦相似度。
 *
 * 为什么需要 L2 归一化：
 * - 向量索引搜索时只做点积（比余弦多一步除法）
 * - 很多向量数据库要求预归一化
 * - 嵌入模型（如 text-embedding-3）输出默认已归一化
 * - 但我们不依赖这个假设，显示归一化确保正确性
 */
export function l2Normalize(vec: number[]): number[] {
  let sum = 0;
  for (const v of vec) sum += v * v;
  const norm = Math.sqrt(sum);
  if (norm === 0) return vec; // 零向量保持原样
  return vec.map((v) => v / norm);
}

// ─── OpenAI Embedding Provider ──────────────────────────────────

/**
 * OpenAI Embedding Provider 实现
 *
 * 调用 OpenAI 的 Embeddings API 将文本转为向量。
 *
 * 使用方式：
 * ```typescript
 * const embedder = new OpenAIEmbeddingProvider({
 *   apiKey: process.env.OPENAI_API_KEY,
 *   dimensions: 384,
 * });
 *
 * const vector = await embedder.embed('用户喜欢吃辣');
 * // vector → [0.023, -0.015, 0.042, ...] 共 384 维
 * ```
 *
 * 为什么默认 384 维：
 * - text-embedding-3-small 支持截断到任意 ≤1536 的维度
 * - 384 是性能和精度的 sweet spot
 * - 更低的维度意味着更快的计算和更少的内存
 * - 与 Jina Embeddings v2 的默认维度一致，方便切换
 */
export class OpenAIEmbeddingProvider implements EmbeddingProvider {
  readonly name = 'openai-embedding';

  private config: Required<OpenAIEmbeddingProviderConfig>;

  constructor(config?: OpenAIEmbeddingProviderConfig) {
    this.config = {
      apiKey: config?.apiKey ?? process.env['OPENAI_API_KEY'] ?? '',
      baseUrl: config?.baseUrl ?? 'https://api.openai.com/v1',
      model: config?.model ?? EMBEDDING_DEFAULTS.model,
      dimensions: config?.dimensions ?? EMBEDDING_DEFAULTS.dimensions,
      timeout: config?.timeout ?? EMBEDDING_DEFAULTS.timeout,
      maxRetries: config?.maxRetries ?? EMBEDDING_DEFAULTS.maxRetries,
    };

    if (!this.config.apiKey) {
      console.warn(
        '[OpenAIEmbeddingProvider] No API key configured. ' +
        'Set OPENAI_API_KEY environment variable or pass apiKey in config.'
      );
    }
  }

  /**
   * 将单段文本转为向量
   *
   * @param text - 输入文本
   * @returns 向量数组（维度 = config.dimensions）
   */
  async embed(text: string): Promise<number[]> {
    const results = await this.embedMany([text]);
    return results[0];
  }

  /**
   * 批量将多段文本转为向量
   *
   * 批量比逐个调用更高效：
   * - 一次 HTTP 请求，减少网络往返
   * - OpenAI API 内部可以并行计算
   * - 减少 API rate limit 消耗
   *
   * @param texts - 输入文本数组
   * @returns 向量数组，与输入顺序一一对应
   */
  async embedMany(texts: string[]): Promise<number[][]> {
    if (texts.length === 0) return [];

    let lastError: Error | null = null;

    for (let attempt = 0; attempt <= this.config.maxRetries; attempt++) {
      try {
        const response = await this.sendRequest(texts);
        return this.parseResponse(response);
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));

        if (attempt >= this.config.maxRetries) break;

        const delay = 1000 * Math.pow(2, attempt) + Math.random() * 1000;
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }

    throw lastError ?? new EmbeddingError('Embedding request failed after retries');
  }

  /**
   * 发送 Embedding API 请求
   */
  private async sendRequest(
    texts: string[]
  ): Promise<OpenAIEmbeddingResponse> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.config.timeout);

    try {
      const response = await fetch(
        `${this.config.baseUrl}/embeddings`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.config.apiKey}`,
          },
          body: JSON.stringify({
            model: this.config.model,
            input: texts,
            dimensions: this.config.dimensions,
          }),
          signal: controller.signal,
        }
      );

      if (!response.ok) {
        const errorText = await response.text().catch(() => '');
        throw new EmbeddingError(
          `HTTP ${response.status}: ${errorText}`,
          response.status
        );
      }

      return (await response.json()) as OpenAIEmbeddingResponse;
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * 解析 API 响应，按输入索引顺序返回向量
   */
  private parseResponse(
    response: OpenAIEmbeddingResponse
  ): number[][] {
    // data 数组按 index 排好序，但防御性排序一下
    const sorted = [...response.data].sort(
      (a, b) => a.index - b.index
    );

    return sorted.map((d) => l2Normalize(d.embedding));
  }
}

// ─── 空实现（测试 / 离线场景） ──────────────────────────────────

/**
 * NoopEmbeddingProvider —— 不真的计算 embedding 的空实现
 *
 * 适用于：
 * - 单元测试中作为占位
 * - 离线环境下的开发
 * - 单纯测试 keyword search 时避免 API 调用
 *
 * 遵循 Null Object Pattern。
 */
export class NoopEmbeddingProvider implements EmbeddingProvider {
  readonly name = 'noop';

  async embed(_text: string): Promise<number[]> {
    return [];
  }

  async embedMany(texts: string[]): Promise<number[][]> {
    return texts.map(() => []);
  }
}
