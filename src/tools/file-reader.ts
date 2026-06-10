/**
 * 文件读取工具 — 在受控范围内读取文件内容
 *
 * 设计决策：
 * - 沙箱路径限制：只允许读取工作目录下的文件
 * - 大小限制：防止读取大文件导致 token 爆炸
 * - 二进制文件检测：非文本文件返回错误提示
 * - 路径遍历防护：对 ../ 等攻击模式做检测
 *
 * 安全是第一优先级。Agent 环境下工具不能完全信任 LLM 传来的参数。
 */

import { z } from 'zod';
import * as fs from 'node:fs/promises';
import * as path from 'node:path';
import type { Tool, ToolMetadata, ToolResult, ToolContext } from '../core/tool.js';

const DEFAULT_MAX_SIZE = 1024 * 64; // 64KB
const DEFAULT_WORK_DIR = process.cwd();

const fileReaderMetadata: ToolMetadata = {
  name: 'file_reader',
  description: '读取指定文件的内容。只能读取工作目录下的文本文件，支持 .md、.ts、.js、.json、.txt、.csv 等格式。',
  parameters: z.object({
    filePath: z.string().describe('要读取的文件路径（相对于工作目录）'),
    maxSize: z.number().int().positive().optional().describe('最大读取字节数（默认 64KB）'),
    encoding: z.enum(['utf-8', 'utf-16le']).optional().describe('文件编码（默认 utf-8）'),
  }),
};

/** 检测是否是二进制文件的前几个字节 */
const BINARY_PATTERNS = [
  Buffer.from([0xff, 0xd8]), // JPEG
  Buffer.from([0x89, 0x50]), // PNG
  Buffer.from([0x47, 0x49]), // GIF
  Buffer.from([0x50, 0x4b]), // ZIP/DOCX
  Buffer.from([0x42, 0x4d]), // BMP
  Buffer.from([0x25, 0x50]), // PDF
];

function isBinaryFile(buf: Buffer): boolean {
  // 检查文件头魔数
  for (const pattern of BINARY_PATTERNS) {
    if (buf.length >= pattern.length && buf.slice(0, pattern.length).equals(pattern)) {
      return true;
    }
  }
  // 检测 NUL 字节（二进制文件的典型特征）
  return buf.includes(0);
}

function isPathTraversal(filePath: string): boolean {
  // 检测路径遍历攻击
  // 用 split 分割路径段，逐段检查 '..'，避免 "test..txt" 误判
  const normalized = path.normalize(filePath);
  const segments = normalized.split(path.sep);
  return (
    segments.includes('..') ||
    path.isAbsolute(normalized)
  );
}

export class FileReaderTool implements Tool {
  metadata = fileReaderMetadata;

  constructor(private options?: { workDir?: string; maxSize?: number }) {}

  async execute(
    args: Record<string, unknown>,
    _ctx?: ToolContext
  ): Promise<ToolResult> {
    const { filePath, maxSize, encoding } = args as {
      filePath: string;
      maxSize?: number;
      encoding?: BufferEncoding;
    };

    if (!filePath || typeof filePath !== 'string') {
      return {
        success: false,
        output: 'Error: filePath is required and must be a string.',
        error: 'INVALID_ARGUMENT',
      };
    }

    // 检查路径遍历
    if (isPathTraversal(filePath)) {
      return {
        success: false,
        output: 'Error: Path traversal is not allowed.',
        error: 'SECURITY_VIOLATION',
      };
    }

    const resolved = path.join(
      this.options?.workDir ?? DEFAULT_WORK_DIR,
      filePath
    );
    const sizeLimit = maxSize ?? this.options?.maxSize ?? DEFAULT_MAX_SIZE;

    try {
      // 检查文件是否存在
      await fs.access(resolved);

      // 检查文件大小
      const stat = await fs.stat(resolved);
      if (stat.size > sizeLimit) {
        return {
          success: false,
          output: `Error: File size (${stat.size} bytes) exceeds limit (${sizeLimit} bytes).`,
          error: 'SIZE_EXCEEDED',
        };
      }

      // 读取文件并检测是否二进制
      const buf = await fs.readFile(resolved);

      if (isBinaryFile(buf)) {
        return {
          success: false,
          output: `Error: File '${filePath}' appears to be a binary file. This tool only supports text files.`,
          error: 'BINARY_FILE',
        };
      }

      const content = buf.toString(encoding ?? 'utf-8');

      return {
        success: true,
        output: content,
        data: {
          fileName: path.basename(filePath),
          size: stat.size,
          lines: content.split('\n').length,
        },
      };
    } catch (err) {
      if ((err as NodeJS.ErrnoException).code === 'ENOENT') {
        return {
          success: false,
          output: `Error: File '${filePath}' not found.`,
          error: 'FILE_NOT_FOUND',
        };
      }
      if ((err as NodeJS.ErrnoException).code === 'EACCES') {
        return {
          success: false,
          output: `Error: Permission denied for file '${filePath}'.`,
          error: 'PERMISSION_DENIED',
        };
      }
      return {
        success: false,
        output: `Error reading file: ${err instanceof Error ? err.message : String(err)}`,
        error: 'IO_ERROR',
      };
    }
  }
}
