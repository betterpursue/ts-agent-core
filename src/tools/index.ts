/**
 * 工具模块入口
 *
 * 导出所有内置工具和辅助函数。
 * 外部使用方式：
 *   import { CalculatorTool, FileReaderTool, executeToolsParallel } from './tools/index.js';
 */

export { CalculatorTool } from './calculator.js';
export { FileReaderTool } from './file-reader.js';
export {
  executeToolsParallel,
  ToolNotFoundError,
  TimeoutError,
  toolMetadataToJsonSchema,
} from './parallel-executor.js';
export type {
  ParallelExecutionOptions,
  ParallelExecutionResult,
} from './parallel-executor.js';
