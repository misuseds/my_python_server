# MCP AI Caller 系统架构总结

## 概述

MCP AI Caller 是一个智能AI助手系统，支持工具调用、记忆管理、多模态交互等功能。

## 系统架构

```
mcp_cline/
├── mcp_ai_caller.py          # 主应用程序（PyQt6界面）
├── mcp_client.py              # MCP客户端（连接所有工具服务器）
├── components/
│   ├── tool_loader.py         # 工具加载器
│   ├── tools_dialog.py        # 工具对话框
│   ├── content_extractor.py   # 内容提取器
│   └── monitoring_window.py  # 监控窗口
└── knowledge.txt              # 流程知识库

tools/
├── blender_cline/             # Blender工具服务器
├── ue_cline/                 # Unreal Engine工具服务器
├── browser_cline/             # 浏览器工具服务器
├── computer_cline/            # 计算机控制工具服务器
├── ocr/                     # OCR工具服务器
├── yolo/                    # 点赞收藏检测工具服务器
├── memory_tool/               # 记忆管理工具服务器 ✅ 新增
│   ├── memory_manager.py      # 记忆管理核心
│   ├── memory_api_tool.py     # MCP服务器接口
│   ├── README.md             # 使用文档
│   └── test_memory.py         # 测试脚本
├── cad_server/               # CAD工具服务器
├── ezdxf_server/            # DXF工具服务器
├── excel_server/             # Excel工具服务器
├── pdf_server/               # PDF工具服务器
├── file_server/              # 文件工具服务器
└── ...

llm_server/
└── llm_class.py              # LLM/VLM服务
```

## 核心功能模块

### 1. 智能工具调用系统

#### 工具状态管理
- **初始状态 (initial)**: 不传递完整工具描述，只提供简短摘要
- **工具状态 (active)**: 传递完整工具列表，支持直接调用

#### 状态转换检测
- **进入工具状态**: 检测到"需要调用工具"、"使用工具"等关键词
- **退出工具状态**: 检测到"任务完成"、"操作完成"等关键词

#### 保护机制
- 最大连续工具调用: 5次
- 重复工具调用检测
- 最大处理轮数: 10轮

### 2. 工具策略控制

#### 策略类型
- **全局允许列表 (allow)**: 只允许指定工具
- **全局拒绝列表 (deny)**: 禁止指定工具（优先级最高）
- **提供商策略 (by_provider)**: 按服务器设置策略
- **沙箱策略 (sandbox)**: 沙箱环境下的额外限制

#### 策略优先级
```
1. 全局拒绝列表 (deny) - 最高优先级
2. 提供商拒绝列表 (by_provider.deny)
3. 提供商允许列表 (by_provider.allow)
4. 全局允许列表 (allow)
5. 沙箱策略 (sandbox)
```

### 3. 记忆管理系统 ✅ 新增

#### 记忆类型
- **普通记忆 (general)**: 存储用户偏好、重要信息、操作结果
- **工具描述记忆 (tool)**: 存储工具使用经验、最佳实践

#### 可用工具
| 工具名 | 功能 | 参数 |
|--------|------|------|
| `read_memory` | 读取记忆文档 | `memory_type` |
| `write_memory` | 写入记忆文档 | `content`, `memory_type`, `append` |
| `search_memory` | 搜索记忆文档 | `query`, `memory_type`, `max_results` |
| `grep_memory` | Grep搜索（正则表达式） | `pattern`, `memory_type`, `context_lines` |
| `clear_memory` | 清空记忆文档 | `memory_type` |
| `get_memory_stats` | 获取统计信息 | `memory_type` |

#### AI记忆使用建议
- 对于重要的信息、用户偏好、操作结果等，可以使用 `write_memory` 写入记忆
- 写入记忆前，先使用 `search_memory` 或 `grep_memory` 检查是否已存在相似信息
- 避免重复写入相同内容
- 普通记忆（general）用于存储用户偏好、重要信息等
- 工具描述记忆（tool）用于存储工具使用经验、最佳实践等

### 4. 事件流处理

#### 事件类型
- **lifecycle**: 生命周期事件 (start | end | error)
- **tool_call_start**: 工具调用开始
- **tool_call_end**: 工具调用结束
- **tool_result**: 工具结果
- **text_delta**: 文本增量
- **text_end**: 文本结束
- **message_end**: 消息结束

#### 事件处理流程
```
用户输入
  ↓
lifecycle: start
  ↓
[检测到工具调用]
  ↓
tool_call_start (工具名, 参数)
  ↓
[执行工具]
  ↓
tool_call_end (工具名, 参数, 结果)
  ↓
tool_result (tool_call_id, 结果)
  ↓
[生成文本回复]
  ↓
text_end (文本内容)
  ↓
message_end (assistant)
  ↓
lifecycle: end
```

### 5. 系统提示管理

#### 系统提示结构
```
## Tool Call Style
- 默认不叙述常规、低风险工具调用
- 只在有帮助时叙述：多步骤工作、复杂问题、敏感操作
- 保持叙述简洁有价值

## 可用工具
- 初始状态：简短描述
- 工具状态：完整描述

## 工具调用规则
1. 初始状态规则
2. 工具状态规则
3. 何时调用工具
4. 何时提供文本回复
5. 多步骤任务处理
6. 工具调用注意事项

## 流程知识
- 从 knowledge.txt 读取
```

## 工具服务器列表

### 已配置的工具服务器

| 服务器名 | 描述 | 工具数量 |
|---------|------|---------|
| blender-tool | Blender工具服务器 | 5+ |
| ue-tool | Unreal Engine工具服务器 | 3+ |
| browser-tool | 浏览器工具服务器 | 1+ |
| computer-tool | 计算机控制工具服务器 | 8+ |
| ocr-tool | OCR工具服务器 | 2+ |
| likefavarite-tools | 点赞收藏检测工具服务器 | 1+ |
| memory-tool | 记忆管理工具服务器 | 6 ✅ 新增 |

### 工具服务器功能

#### Blender工具
- 3D模型处理
- 导入导出
- 骨骼绑定
- 材质编辑
- 动画制作

#### UE工具
- FBX导入
- UE启动
- MOD构建
- 场景管理

#### 计算机控制工具
- 鼠标点击
- 键盘输入
- 滚轮滚动
- 截图功能
- 窗口操作

#### OCR工具
- 文字识别
- 坐标定位
- 文本提取

#### 浏览器工具
- 打开URL
- 页面导航
- 内容获取

#### 点赞收藏检测工具
- 检测点赞按钮
- 检测收藏按钮
- 坐标返回

#### 记忆工具 ✅ 新增
- 读取记忆
- 写入记忆
- 搜索记忆
- Grep记忆
- 清空记忆
- 获取统计

## 使用流程

### 用户交互流程

```
用户输入 (/r 命令)
  ↓
AI判断是否需要工具
  ↓
  ├─ 不需要工具 → 直接文本回复
  │   ↓
  │   显示回复
  │   ↓
  │   结束
  │
  └─ 需要工具 → 进入工具状态
      ↓
      AI选择工具
      ↓
      执行工具
      ↓
      获取结果
      ↓
      AI判断是否继续
      ↓
        ├─ 继续工具调用 → 重复执行
        │   ↓
        │   最多5次连续调用
        │
        └─ 任务完成 → 退出工具状态
            ↓
            AI生成总结
            ↓
            显示回复
            ↓
            结束
```

### 记忆使用流程

```
用户提到重要信息
  ↓
AI判断是否需要记录
  ↓
  ├─ 需要记录 → 调用 write_memory
  │   ↓
  │   先使用 search_memory/grep_memory 检查重复
  │   ↓
  │   写入记忆
  │   ↓
  │   确认写入成功
  │
  └─ 不需要记录 → 继续处理
      ↓
      显示回复
```

### 查询历史流程

```
用户询问历史信息
  ↓
AI调用 search_memory 或 grep_memory
  ↓
  搜索记忆文档
  ↓
  返回匹配结果
  ↓
  AI基于结果生成回复
  ↓
  显示回复
```

## 配置文件

### MCP客户端配置 (mcp_client.py)
```python
servers = {
    'blender-tool': {...},
    'ue-tool': {...},
    'browser-tool': {...},
    'computer-tool': {...},
    'ocr-tool': {...},
    'likefavarite-tools': {...},
    'memory-tool': {...}  # ✅ 新增
}
```

### 工具策略配置
```python
tools_policy = {
    'allow': [],           # 允许的工具列表
    'deny': [],            # 禁止的工具列表
    'by_provider': {},      # 按提供商的策略
    'sandbox': {}          # 沙箱策略
}
```

### 记忆目录结构
```
memory/
├── general_memory.txt      # 普通记忆
└── tool_description.txt    # 工具描述记忆
```

## 技术特性

### 1. 异步处理
- 所有工具调用都是异步的
- 支持并行执行多个工具
- 使用asyncio管理事件循环

### 2. 错误处理
- 完善的异常捕获
- 详细的错误日志
- 用户友好的错误提示

### 3. 性能优化
- 工具加载缓存
- 会话池管理
- 按需加载工具描述

### 4. 线程安全
- PyQt信号机制
- 线程间通信
- 主线程UI更新

### 5. 状态管理
- 工具状态自动转换
- 对话历史维护
- 记忆持久化

## 开发指南

### 添加新工具服务器

1. 在 `tools/` 下创建新文件夹
2. 实现MCP服务器接口
3. 在 `mcp_client.py` 中注册服务器
4. 重启应用加载工具

### 扩展记忆功能

1. 修改 `memory_manager.py` 添加新方法
2. 在 `memory_api_tool.py` 中添加新工具
3. 更新 `README.md` 文档
4. 运行测试验证

### 自定义系统提示

1. 修改 `_build_system_prompt` 方法
2. 调整工具描述
3. 更新工具调用规则
4. 测试AI行为

## 测试

### 单元测试
```bash
# 测试记忆管理
cd tools/memory_tool
python test_memory.py

# 测试各个工具服务器
cd tools/blender_cline
python demo.py
```

### 集成测试
1. 启动MCP AI Caller
2. 使用 `/r` 命令测试工具调用
3. 使用 `/h` 命令查看工具列表
4. 测试记忆读写功能

## 最佳实践

### 1. 工具调用
- 避免不必要的工具调用
- 合理使用工具状态
- 及时完成任务退出

### 2. 记忆管理
- 定期清理过时记忆
- 避免重复写入
- 合理分类存储

### 3. 系统提示
- 保持提示简洁清晰
- 提供明确的指导
- 根据状态动态调整

### 4. 错误处理
- 记录详细日志
- 提供用户友好提示
- 优雅降级处理

## 故障排除

### 常见问题

**Q: 工具加载失败？**
A: 检查工具服务器脚本路径，确认Python环境

**Q: 记忆写入失败？**
A: 检查 `memory/` 目录权限，确认磁盘空间

**Q: 工具调用无限循环？**
A: 检查连续工具调用次数，调整策略

**Q: 状态转换不工作？**
A: 检查系统提示中的关键词配置

## 更新日志

### v2.0.0 (2026-01-28)
- ✅ 添加智能工具调用系统（初始/工具状态）
- ✅ 添加工具策略控制（allow/deny/by_provider）
- ✅ 添加事件流处理机制
- ✅ 添加记忆管理系统
- ✅ 改进系统提示（支持动态工具描述）
- ✅ 添加状态转换检测
- ✅ 添加保护机制（连续调用、重复检测）
- ✅ 完善错误处理和日志

### v1.0.0 (初始版本)
- 基础MCP工具调用功能
- 工具加载和管理
- PyQt6界面
- VLM集成

## 总结

MCP AI Caller 现在是一个功能完整的智能AI助手系统，具备：

1. ✅ 智能工具调用（状态管理、动态描述）
2. ✅ 工具策略控制（灵活的访问控制）
3. ✅ 记忆管理系统（持久化知识）
4. ✅ 事件流处理（完整的生命周期追踪）
5. ✅ 多模态支持（文本、图像、工具）
6. ✅ 错误处理（完善的异常管理）
7. ✅ 性能优化（缓存、异步、按需加载）

系统已经准备好投入使用！
