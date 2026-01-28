# 记忆系统使用说明

## 概述

记忆系统为AI提供了读写持久化记忆的能力，支持普通记忆和工具描述记忆两种类型。

## 功能特性

### 1. 记忆类型

- **普通记忆 (general)**: 存储用户偏好、重要信息、操作结果等
- **工具描述记忆 (tool)**: 存储工具使用经验、最佳实践等

### 2. 可用工具

#### read_memory
读取记忆文档

**参数：**
- `memory_type` (可选): 记忆类型，'general' 或 'tool'，默认 'general'

**示例：**
```python
# 读取普通记忆
read_memory(memory_type='general')

# 读取工具描述记忆
read_memory(memory_type='tool')
```

#### write_memory
写入记忆文档（AI可以自主决定是否需要写入）

**参数：**
- `content` (必需): 要写入的内容
- `memory_type` (可选): 记忆类型，'general' 或 'tool'，默认 'general'
- `append` (可选): 是否追加到文件末尾，默认 True

**示例：**
```python
# 追加写入普通记忆
write_memory(content='用户喜欢使用Blender进行3D建模', memory_type='general', append=True)

# 覆盖写入工具描述记忆
write_memory(content='Blender导入FBX的最佳实践：先检查单位设置', memory_type='tool', append=False)
```

#### search_memory
搜索记忆文档（支持关键词搜索）

**参数：**
- `query` (必需): 搜索关键词
- `memory_type` (可选): 记忆类型，'general' 或 'tool'，默认 'general'
- `max_results` (可选): 最大返回结果数，默认 5

**示例：**
```python
# 搜索普通记忆
search_memory(query='Blender', memory_type='general', max_results=5)

# 搜索工具描述记忆
search_memory(query='导入', memory_type='tool', max_results=3)
```

#### grep_memory
在记忆文档中搜索模式（类似grep命令，支持正则表达式）

**参数：**
- `pattern` (必需): 正则表达式模式
- `memory_type` (可选): 记忆类型，'general' 或 'tool'，默认 'general'
- `context_lines` (可选): 上下文行数，默认 2

**示例：**
```python
# 搜索包含"用户"和"喜欢"的行
grep_memory(pattern='用户.*喜欢', memory_type='general', context_lines=2)

# 搜索包含"Blender"的行
grep_memory(pattern='Blender', memory_type='tool', context_lines=1)
```

#### clear_memory
清空记忆文档（谨慎使用，会删除所有内容）

**参数：**
- `memory_type` (可选): 记忆类型，'general' 或 'tool'，默认 'general'

**示例：**
```python
# 清空普通记忆
clear_memory(memory_type='general')

# 清空工具描述记忆
clear_memory(memory_type='tool')
```

#### get_memory_stats
获取记忆统计信息

**参数：**
- `memory_type` (可选): 记忆类型，'general' 或 'tool'，默认 'general'

**返回信息：**
- `exists`: 文件是否存在
- `size`: 文件大小（字节）
- `lines`: 行数
- `last_modified`: 最后修改时间

**示例：**
```python
# 获取普通记忆统计
get_memory_stats(memory_type='general')

# 获取工具描述记忆统计
get_memory_stats(memory_type='tool')
```

## 文件结构

```
memory/
├── general_memory.txt      # 普通记忆文档
└── tool_description.txt    # 工具描述记忆文档
```

## AI使用建议

### 何时写入记忆

**建议写入的情况：**
- 用户明确表示偏好或习惯
- 重要的操作结果或结论
- 工具使用的最佳实践
- 需要跨会话保持的信息
- 用户的问题和解决方案

**不建议写入的情况：**
- 临时信息或一次性查询
- 重复的常见知识
- 敏感信息（密码、密钥等）
- 过于琐碎的内容

### 写入前的检查

在写入记忆前，建议：
1. 使用 `search_memory` 或 `grep_memory` 检查是否已存在相似信息
2. 避免重复写入相同内容
3. 确认信息确实值得持久化
4. 选择合适的记忆类型（general 或 tool）

### 记忆内容格式建议

**普通记忆格式：**
```
--- 2026-01-28 10:30:15 ---
用户偏好：使用Blender进行3D建模，偏好使用快捷键提高效率
```

**工具描述记忆格式：**
```
--- 2026-01-28 10:30:15 ---
Blender导入FBX最佳实践：
1. 先检查单位设置（米 vs 厘米）
2. 确认轴心位置
3. 检查材质路径是否正确
```

## 使用示例

### 场景1：记录用户偏好

```
用户：我喜欢用Blender做3D建模
AI：[调用 write_memory]
AI：已记录您的偏好到记忆中
```

### 场景2：查询历史信息

```
用户：我之前说过喜欢用什么软件吗？
AI：[调用 search_memory，query='喜欢']
AI：根据记忆，您喜欢使用Blender进行3D建模
```

### 场景3：记录工具使用经验

```
用户：这个工具怎么用最好？
AI：[调用 grep_memory，pattern='工具.*使用']
AI：根据记忆，这个工具的最佳实践是...
AI：[调用 write_memory，content='工具使用经验总结...']
```

### 场景4：跨会话保持信息

```
会话1：
用户：记住这个设置：分辨率1920x1080
AI：[调用 write_memory]
AI：已记录设置到记忆

会话2（重启后）：
用户：我之前设置的分辨率是多少？
AI：[调用 search_memory，query='分辨率']
AI：根据记忆，您设置的分辨率是1920x1080
```

## 技术细节

### 记忆管理器类

`MemoryManager` 类提供以下方法：

- `read_memory(memory_type)`: 读取记忆
- `write_memory(content, memory_type, append)`: 写入记忆
- `search_memory(query, memory_type, max_results)`: 搜索记忆
- `grep_memory(pattern, memory_type, context_lines)`: Grep搜索
- `clear_memory(memory_type)`: 清空记忆
- `get_memory_stats(memory_type)`: 获取统计信息

### 搜索算法

- **关键词搜索**: 基于字符串匹配，计算匹配分数
- **Grep搜索**: 支持正则表达式，返回匹配行和上下文
- **匹配分数**: 0-1之间，1表示完全匹配

### 文件操作

- 所有文件使用UTF-8编码
- 写入操作自动添加时间戳
- 追加模式在文件末尾添加分隔符
- 支持覆盖模式（append=False）

## 配置

### 记忆目录

记忆目录默认位于：
```
{项目根目录}/memory/
```

可以通过修改 `MemoryManager` 的 `base_dir` 参数来更改。

### MCP服务器配置

记忆工具已注册到MCP客户端，配置文件：
```python
'memory-tool': {
    'command': sys.executable,
    'args': ['path/to/memory_api_tool.py'],
    'description': '记忆管理工具服务器'
}
```

## 故障排除

### 常见问题

**Q: 记忆文件在哪里？**
A: 默认在 `{项目根目录}/memory/` 目录下

**Q: 如何清空记忆？**
A: 使用 `clear_memory` 工具，指定 `memory_type` 参数

**Q: 搜索不到结果怎么办？**
A: 尝试使用 `grep_memory` 工具，使用更灵活的正则表达式

**Q: 记忆文件太大怎么办？**
A: 可以使用 `get_memory_stats` 查看统计信息，然后使用 `clear_memory` 清空

**Q: 如何备份记忆？**
A: 直接复制 `memory/` 目录即可备份

## 最佳实践

1. **定期清理**: 定期检查记忆内容，删除过时或重复的信息
2. **合理分类**: 普通信息和工具描述分开存储
3. **格式统一**: 使用一致的格式，便于搜索和维护
4. **避免冗余**: 写入前先搜索，避免重复
5. **敏感信息**: 不要存储密码、密钥等敏感信息
6. **时间戳**: 利用自动时间戳，便于追溯和清理

## 更新日志

### v1.0.0 (2026-01-28)
- 初始版本
- 实现6个记忆工具
- 支持普通记忆和工具描述记忆
- 支持关键词搜索和Grep搜索
- 支持统计信息查询
