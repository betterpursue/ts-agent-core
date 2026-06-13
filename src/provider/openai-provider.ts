/**
 * OpenAI Provider 实现
 *
 * 真实的 OpenAI API 调用器。连接我们的消息模型和 OpenAI 的 Chat Completions API。
 *
 * 设计原则：
 * - 不引入 OpenAI SDK 依赖：用原生 fetch 调用 API，保持零外部依赖
 * - 格式转换通过 provider-adapter 层，职责分离
 * - API key 从环境变量读取，不硬编码
 * - 超时、重试、错误处理一层清
 * - Token 用量统计从 API 响应中提取
 *
 * 使用方式：
 * ```typescript
 * const provider = new OpenAIProvider({
 *   apiKey: process.env.OPENAI_API_KEY,
 *   baseUrl: 'https://api.openai.com/v1',
 * });
 * ```
 */

import type { LLMProvider, LLMRequest, LLMResponse, LLMChunk } from '../core/llm.js';
import { toOpenAIMessages, fromOpenAIToolCalls } from '../core/provider-adapter.js';
import { buildOpenAITools, type OpenAITool } from './schema-builder.js';

// ─── 配置 ───────────────────────────────────────────────────────

export interface OpenAIProviderConfig {
  /** OpenAI API Key（默认从 OPENAI_API_KEY 环境变量读取） */
  apiKey?: string;
  /** API 基础 URL（默认 https://api.openai.com/v1） */
  baseUrl?: string;
  /** 请求超时 ms（默认 60000） */
  timeout?: number;
  /** 最大重试次数（默认 2） */
  maxRetries?: number;
  /** 重试间隔基数 ms（默认 1000，每次重试翻倍） */
  retryBaseDelay?: number;
  /** 默认模型（每次请求可覆盖） */
  defaultModel?: string;
  /** 自定义请求头 */
  headers?: Record<string, string>;
}

// ─── API 响应类型 ──────────────────────────────────────────────

interface OpenAICompletionResponse {
  id: string;
  object: 'chat.completion';
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: 'assistant';
      content: string | null;
      tool_calls?: Array<{
        id: string;
        type: 'function';
        function: {
          name: string;
          arguments: string;
        };
      }>;
    };
    finish_reason: 'stop' | 'tool_calls' | 'length' | 'content_filter' | null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

interface OpenAIStreamChunk {
  choices: Array<{
    delta: {
      role?: string;
      content?: string | null;
      tool_calls?: Array<{
        index: number;
        id?: string;
        type?: 'function';
        function?: {
          name?: string;
          arguments?: string;
        };
      }>;
    };
    finish_reason?: 'stop' | 'tool_calls' | 'length' | null;
  }>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
}

// ─── 错误处理 ───────────────────────────────────────────────────

export class OpenAIError extends Error {
  constructor(
    message: string,
    public readonly statusCode?: number,
    public readonly code?: string
  ) {
    super(message);
    this.name = 'OpenAIError';
  }
}

export class OpenAIRateLimitError extends OpenAIError {
  constructor(retryAfter?: number) {
    super(
      `Rate limited${retryAfter ? `, retry after ${retryAfter}s` : ''}`,
      429,
      'rate_limit_exceeded'
    );
    this.name = 'OpenAIRateLimitError';
  }
}

export class OpenAIAuthError extends OpenAIError {
  constructor(message: string) {
    super(message, 401, 'authentication_error');
    this.name = 'OpenAIAuthError';
  }
}

// ─── Provider 实现 ──────────────────────────────────────────────

export class OpenAIProvider implements LLMProvider {
  readonly name = 'openai';

  private config: Required<OpenAIProviderConfig>;

  constructor(config?: OpenAIProviderConfig) {
    this.config = {
      apiKey: config?.apiKey ?? process.env['OPENAI_API_KEY'] ?? '',
      baseUrl: config?.baseUrl ?? 'https://api.openai.com/v1',
      timeout: config?.timeout ?? 60_000,
      maxRetries: config?.maxRetries ?? 2,
      retryBaseDelay: config?.retryBaseDelay ?? 1_000,
      defaultModel: config?.defaultModel ?? 'gpt-4o',
      headers: config?.headers ?? {},
    };

    if (!this.config.apiKey) {
      console.warn(
        '[OpenAIProvider] No API key configured. ' +
        'Set OPENAI_API_KEY environment variable or pass apiKey in config.'
      );
    }
  }

  async complete(request: LLMRequest): Promise<LLMResponse> {
    // 1. 构建请求体
    const body = this.buildRequestBody(request);
    let lastError: Error | null = null;

    // 2. 带重试的请求
    for (let attempt = 0; attempt <= this.config.maxRetries; attempt++) {
      try {
        const response = await this.sendRequest(body);
        return this.parseResponse(response);
      } catch (err) {
        lastError = err instanceof Error ? err : new Error(String(err));

        // 不可重试的错误直接抛出
        if (err instanceof OpenAIAuthError) throw err;

        // 如果是最后一个 attempt，不再重试
        if (attempt >= this.config.maxRetries) break;

        // 重试等待（指数退避 + 随机抖动）
        const delay =
          this.config.retryBaseDelay * Math.pow(2, attempt) +
          Math.random() * 1000;
        await new Promise((resolve) => setTimeout(resolve, delay));
      }
    }

    throw lastError ?? new OpenAIError('Request failed after retries');
  }

  async completeStream(
    request: LLMRequest,
    onChunk: (chunk: LLMChunk) => void
  ): Promise<LLMResponse> {
    const body = this.buildRequestBody(request, true);
    let accumulatedContent = '';
    let accumulatedToolCalls: Array<{
      index: number;
      id?: string;
      name?: string;
      args: string;
    }> = [];
    let finishReason: LLMResponse['finishReason'] = 'stop';

    try {
      const response = await this.sendStreamRequest(body);

      if (!response.body) {
        throw new OpenAIError('Stream response has no body');
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? ''; // 不完整行留在 buffer 中

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith('data: ')) continue;

          const data = trimmed.slice(6); // 去掉 "data: " 前缀
          if (data === '[DONE]') continue;

          try {
            const chunk = JSON.parse(data) as OpenAIStreamChunk;
            const delta = chunk.choices?.[0]?.delta;
            const finish = chunk.choices?.[0]?.finish_reason;

            if (!delta && !finish) continue;

            const hanakoChunk: LLMChunk = {};

            // 处理内容
            if (delta?.content) {
              hanakoChunk.contentDelta = delta.content;
              accumulatedContent += delta.content;
            }

            // 处理 tool_calls（流式）
            if (delta?.tool_calls) {
              for (const tc of delta.tool_calls) {
                let existing = accumulatedToolCalls[tc.index];
                if (!existing) {
                  existing = { index: tc.index, args: '' };
                  accumulatedToolCalls[tc.index] = existing;
                }
                if (tc.id) existing.id = tc.id;
                if (tc.function?.name) existing.name = tc.function.name;
                if (tc.function?.arguments) {
                  existing.args += tc.function.arguments;
                }
              }

              hanakoChunk.toolCallDelta = {
                id: delta.tool_calls[0]?.id,
                name: delta.tool_calls[0]?.function?.name,
                args: delta.tool_calls[0]?.function?.arguments,
              };
            }

            // 处理结束原因
            if (finish) {
              switch (finish) {
                case 'stop':
                  finishReason = 'stop';
                  break;
                case 'tool_calls':
                  finishReason = 'tool_calls';
                  break;
                case 'length':
                  finishReason = 'length';
                  break;
                default:
                  finishReason = 'stop';
              }
              hanakoChunk.finishReason = finishReason;
            }

            onChunk(hanakoChunk);
          } catch {
            // 跳过无法解析的 chunk
          }
        }
      }

      // 组装最终响应
      const toolCalls = accumulatedToolCalls
        .filter((tc) => tc.id && tc.name)
        .map((tc) => {
          let args: Record<string, unknown> = {};
          try {
            args = JSON.parse(tc.args) as Record<string, unknown>;
          } catch {
            // args 可能在流式传输中不完整，保留空对象
            // 后续 ResponseParser 会做修复
          }
          return {
            id: tc.id!,
            name: tc.name!,
            args,
          };
        });

      return {
        content: accumulatedContent,
        toolCalls: toolCalls.length > 0 ? toolCalls : undefined,
        finishReason,
      };
    } catch (err) {
      throw new OpenAIError(
        `Stream error: ${err instanceof Error ? err.message : String(err)}`
      );
    }
  }

  // ─── 内部方法 ──────────────────────────────────────────────

  /**
   * 构建 OpenAI API 请求体
   */
  private buildRequestBody(
    request: LLMRequest,
    stream = false
  ): Record<string, unknown> {
    const openaiMessages = toOpenAIMessages(request.messages);

    const body: Record<string, unknown> = {
      model: request.model ?? this.config.defaultModel,
      messages: openaiMessages,
      stream,
    };

    // 添加 Tool 描述
    if (request.tools && request.tools.length > 0) {
      body['tools'] = buildOpenAITools(request.tools);

      // tool_choice
      if (request.toolChoice) {
        body['tool_choice'] = request.toolChoice;
      }
    }

    // 可选参数
    if (request.maxTokens !== undefined) body['max_tokens'] = request.maxTokens;
    if (request.temperature !== undefined) body['temperature'] = request.temperature;

    // 透传厂商特定参数
    if (request.extraParams) {
      Object.assign(body, request.extraParams);
    }

    return body;
  }

  /**
   * 发送 HTTP 请求到 OpenAI API
   */
  private async sendRequest(
    body: Record<string, unknown>
  ): Promise<OpenAICompletionResponse> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.config.timeout);

    try {
      const response = await fetch(
        `${this.config.baseUrl}/chat/completions`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.config.apiKey}`,
            ...this.config.headers,
          },
          body: JSON.stringify(body),
          signal: controller.signal,
        }
      );

      if (!response.ok) {
        await this.handleErrorResponse(response);
      }

      return (await response.json()) as OpenAICompletionResponse;
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * 发送流式请求
   */
  private async sendStreamRequest(
    body: Record<string, unknown>
  ): Promise<Response> {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), this.config.timeout);

    try {
      const response = await fetch(
        `${this.config.baseUrl}/chat/completions`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.config.apiKey}`,
            'Accept': 'text/event-stream',
            ...this.config.headers,
          },
          body: JSON.stringify(body),
          signal: controller.signal,
        }
      );

      if (!response.ok) {
        await this.handleErrorResponse(response);
      }

      return response;
    } finally {
      clearTimeout(timer);
    }
  }

  /**
   * 将 OpenAI 的响应解析为内部 LLMResponse
   */
  private parseResponse(response: OpenAICompletionResponse): LLMResponse {
    const choice = response.choices?.[0];
    if (!choice) {
      throw new OpenAIError('Empty response: no choices returned');
    }

    const rawMessage = choice.message;
    const finishReason: LLMResponse['finishReason'] = this.mapFinishReason(
      choice.finish_reason
    );

    // 转换 tool_calls
    const toolCalls = rawMessage.tool_calls
      ? fromOpenAIToolCalls(
          rawMessage.tool_calls as Array<{
            id: string;
            type: 'function';
            function: { name: string; arguments: string };
          }>
        )
      : undefined;

    return {
      content: rawMessage.content ?? '',
      toolCalls,
      finishReason,
      usage: response.usage
        ? {
            promptTokens: response.usage.prompt_tokens,
            completionTokens: response.usage.completion_tokens,
            totalTokens: response.usage.total_tokens,
          }
        : undefined,
    };
  }

  /**
   * 映射 OpenAI 的 finish_reason 到内部枚举
   */
  private mapFinishReason(
    reason: string | null
  ): LLMResponse['finishReason'] {
    switch (reason) {
      case 'stop':
        return 'stop';
      case 'tool_calls':
        return 'tool_calls';
      case 'length':
        return 'length';
      case 'content_filter':
        return 'error';
      default:
        return 'stop';
    }
  }

  /**
   * 处理错误响应
   */
  private async handleErrorResponse(
    response: Response
  ): Promise<never> {
    let errorBody: Record<string, unknown> | null = null;
    try {
      errorBody = (await response.json()) as Record<string, unknown>;
    } catch {
      // 忽略解析失败
    }

    const errorMessage =
      (errorBody?.['error'] as { message?: string })?.message ??
      `HTTP ${response.status}`;

    switch (response.status) {
      case 401:
        throw new OpenAIAuthError(errorMessage);
      case 429: {
        const retryAfter = response.headers.get('retry-after');
        throw new OpenAIRateLimitError(
          retryAfter ? parseInt(retryAfter, 10) : undefined
        );
      }
      default:
        throw new OpenAIError(errorMessage, response.status);
    }
  }
}
