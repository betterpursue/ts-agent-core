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
  TimeoutError,
  hasConflict,
  buildLayers,
} from './parallel-executor.js';
export type {
  ParallelExecutionResult,
  ToolCallRequest,
} from './parallel-executor.js';
export type { ParallelExecutionOptions } from '../core/agent.js';
