# RAG 智能检索服务 API 文档

## 1. 服务概述

### 1.1 服务简介
RAG (Retrieval-Augmented Generation) 智能检索服务是一个基于向量数据库和大语言模型的智能问答系统，支持自然语言查询、流式输出、多文档检索等功能。

### 1.2 技术栈
- **后端框架**: FastAPI
- **向量数据库**: Milvus
- **大语言模型**: 通义千问 (DashScope)
- **向量嵌入**: 通义千问文本嵌入模型
- **WebSocket**: 实时流式输出

### 1.3 服务地址
- **API 基础路径**: `http://localhost:8088`
- **WebSocket 基础路径**: `ws://localhost:8088`

## 2. 认证机制

### 2.1 认证方式
使用 HTTP Bearer Token 进行认证，Token 需包含在请求头中：
```
Authorization: Bearer your_token_here
```

### 2.2 必需请求头
| 字段名 | 类型 | 描述 | 示例值 |
|--------|------|------|--------|
| X-User-Id | string | 用户唯一标识 | "1" |
| X-Department | string | 用户所属部门 | "market" |
| Authorization | string | 认证令牌 | "Bearer your_token_here" |

## 3. API 接口

### 3.1 智能检索接口

#### 3.1.1 普通检索接口

**请求信息**
- **URL**: `/query`
- **方法**: `POST`
- **Content-Type**: `application/json`

**请求体**
```json
{
  "text": "帮我找小鼠心脏的数据",
  "use_llm": true
}
```

**请求参数说明**
| 参数名 | 类型 | 必填 | 描述 | 示例值 |
|--------|------|------|------|--------|
| text | string | 是 | 用户自然语言查询 | "帮我找小鼠心脏的数据" |
| use_llm | boolean | 否 | 是否需要LLM生成回答 | true |

**响应格式**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "answer": "生成的回答内容",
    "mode": "semantic",
    "intent": {
      "original_text": "帮我找小鼠心脏的数据",
      "extracted_filters": {"species": "小鼠", "tissue": "心脏"},
      "search_term": "小鼠心脏数据"
    },
    "sources": [
      {
        "node_id": "node123",
        "text": "相关文本内容",
        "score": 0.95,
        "metadata": {
          "filename": "data.xlsx",
          "species": "小鼠",
          "tissue": "心脏"
        }
      }
    ],
    "all_rows": [/* 所有检索结果 */],
    "total_count": 10
  }
}
```

#### 3.1.2 流式检索接口

**请求信息**
- **URL**: `/query/streaming`
- **方法**: `POST`
- **Content-Type**: `application/json`

**请求体**
```json
{
  "text": "帮我找小鼠心脏的数据",
  "connection_id": "ws-12345",
  "use_llm": true
}
```

**请求参数说明**
| 参数名 | 类型 | 必填 | 描述 | 示例值 |
|--------|------|------|------|--------|
| text | string | 是 | 用户自然语言查询 | "帮我找小鼠心脏的数据" |
| connection_id | string | 是 | WebSocket连接ID | "ws-12345" |
| use_llm | boolean | 否 | 是否需要LLM生成回答 | true |

**响应格式**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "mode": "semantic",
    "intent": {
      "original_text": "帮我找小鼠心脏的数据",
      "extracted_filters": {"species": "小鼠", "tissue": "心脏"},
      "search_term": "小鼠心脏数据"
    },
    "sources": [/* 相关片段 */],
    "all_rows": [/* 所有检索结果 */],
    "total_count": 10
  }
}
```

### 3.2 文档上传接口

**请求信息**
- **URL**: `/documents/upload`
- **方法**: `POST`
- **Content-Type**: `multipart/form-data`

**请求参数**
| 参数名 | 类型 | 必填 | 描述 | 示例值 |
|--------|------|------|------|--------|
| file | file | 是 | 要上传的文件 | 二进制文件 |
| target_dept | string | 否 | 目标部门（默认使用用户所属部门） | "market" |

**响应格式**
```json
{
  "code": 200,
  "message": "success",
  "data": {
    "file_id": "uuid-12345",
    "filename": "data.xlsx",
    "department": "market",
    "status": "uploaded"
  }
}
```

### 3.3 健康检查接口

**请求信息**
- **URL**: `/health`
- **方法**: `GET`

**响应格式**
```json
{
  "status": "ok",
  "module": "rag-service"
}
```

## 4. WebSocket 协议

### 4.1 WebSocket 连接

**连接地址**
- **URL**: `/ws/chat`

**连接流程**
1. 客户端建立 WebSocket 连接
2. 服务端返回连接信息
3. 客户端调用流式检索接口
4. 服务端通过 WebSocket 推送流式内容

### 4.2 WebSocket 消息格式

#### 4.2.1 连接信息 (服务端 → 客户端)
```json
{
  "type": "connection_info",
  "connection_id": "ws-12345"
}
```

#### 4.2.2 流式内容 (服务端 → 客户端)
```json
{
  "type": "content",
  "value": "生成的文本片段"
}
```

#### 4.2.3 流式结束 (服务端 → 客户端)
```json
{
  "type": "end_of_stream",
  "status": "success"
}
```

#### 4.2.4 心跳检测 (客户端 ↔ 服务端)
```json
// 客户端 → 服务端
{"type": "ping"}

// 服务端 → 客户端
{"type": "pong"}
```

## 5. 错误码定义

| 错误码 | 描述 | HTTP状态码 |
|--------|------|------------|
| 200 | 成功 | 200 |
| 400 | 请求参数错误 | 400 |
| 401 | 未认证 | 401 |
| 403 | 权限不足 | 403 |
| 404 | 资源不存在 | 404 |
| 500 | 服务器内部错误 | 500 |
| 501 | 功能未实现 | 501 |
| 503 | 服务不可用 | 503 |

## 6. 测试用例

### 6.1 智能检索测试用例

#### 测试用例 1: 基本检索
**请求**
```json
{
  "text": "帮我找小鼠心脏的注释结果",
  "use_llm": true
}
```

**预期结果**
- 返回状态码 200
- 包含相关的注释结果
- 生成合理的LLM回答

#### 测试用例 2: 无LLM检索
**请求**
```json
{
  "text": "帮我找小鼠心脏的数据",
  "use_llm": false
}
```

**预期结果**
- 返回状态码 200
- 包含检索结果
- answer字段为空或提示未生成回答

#### 测试用例 3: 流式检索
**步骤**
1. 建立WebSocket连接
2. 获取connection_id
3. 调用流式检索接口
4. 监听WebSocket消息

**预期结果**
- WebSocket连接成功
- 流式接收LLM生成内容
- 最后收到end_of_stream消息

### 6.2 文档上传测试用例

#### 测试用例 1: 正常上传
**请求**
- 方法: POST
- URL: /documents/upload
- 文件: test.xlsx
- target_dept: market

**预期结果**
- 返回状态码 200
- 返回包含file_id的成功响应

#### 测试用例 2: 上传不支持的文件类型
**请求**
- 方法: POST
- URL: /documents/upload
- 文件: test.exe

**预期结果**
- 返回状态码 400
- 返回不支持的文件类型错误

### 6.3 健康检查测试用例

#### 测试用例 1: 服务正常
**请求**
- 方法: GET
- URL: /health

**预期结果**
- 返回状态码 200
- 返回 {"status": "ok", "module": "rag-service"}

## 7. 前后端联调注意事项

### 7.1 前端注意事项
1. **WebSocket连接管理**
   - 确保WebSocket连接正确关闭，避免资源泄漏
   - 实现心跳机制，保持连接活跃
   - 处理连接断开和重连逻辑

2. **流式输出处理**
   - 使用占位符实时更新内容，避免频繁重绘
   - 考虑使用Markdown渲染，支持富文本输出
   - 实现滚动到底部功能，确保用户看到最新内容

3. **错误处理**
   - 统一处理API错误码
   - 显示友好的错误提示
   - 实现重试机制

### 7.2 后端Java联调注意事项
1. **请求格式**
   - 确保请求头包含所有必需字段
   - 流式请求需先建立WebSocket连接
   - 上传文件使用multipart/form-data格式

2. **响应处理**
   - 统一处理JSON响应
   - 实现异步处理，避免阻塞主线程
   - 考虑使用线程池处理WebSocket消息

3. **安全考虑**
   - 保护Bearer Token，避免泄露
   - 实现请求限流，防止恶意请求
   - 验证请求来源和权限

## 8. 性能优化建议

1. **批量处理**
   - 考虑批量上传文档
   - 实现异步处理，提高并发能力

2. **缓存策略**
   - 缓存热点查询结果
   - 缓存向量索引，加速检索

3. **资源管理**
   - 合理设置连接池大小
   - 实现超时机制，避免长时间阻塞
   - 监控系统资源使用情况

## 9. 版本控制

| 版本 | 日期 | 变更说明 |
|------|------|----------|
| 1.0.0 | 2026-01-20 | 初始版本 |
| 1.1.0 | YYYY-MM-DD | 新增功能说明 |

## 10. 联系方式

- 开发团队: Trae AI
- 维护邮箱: support@trae.ai
- 技术支持: 123-456-7890

---

**文档更新时间**: 2026-01-20
**文档版本**: 1.0.0