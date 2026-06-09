# ts-agent-core

从零搭建 TypeScript Agent 运行时核心 —— 主干项目。

## 定位

这是一个**可扩展的主干项目**。核心模块围绕 interface 设计，后续系列（Redis 模块、MySQL 模块、记忆系统增强等）只需实现对应接口并注册，即可无缝接入。

## 模块结构

```
src/
├── core/          # 核心接口与类型定义
│   ├── agent.ts   # Agent 接口
│   ├── tool.ts    # Tool 接口 + ToolRegistry
│   ├── memory.ts  # 短期/长期记忆接口
│   ├── message.ts # 消息模型
│   ├── session.ts # 会话管理
│   └── llm.ts     # LLM Provider 抽象
├── runloop/       # Agent 主循环实现
├── tools/         # 内置工具实现
├── memory/        # 记忆系统实现
└── index.ts       # 入口
```

## 开发

```bash
npm run build    # 编译 TypeScript
npm run test     # 运行测试
npm run test:watch  # 监听模式
```

## 许可

MIT
