# Milvus Schema 结构分析

## 1. 核心 Schema 定义

根据代码库中的配置和文档，数据存入 Milvus 的 schema 结构如下：

### 1.1 基础字段
| 字段名 | 数据类型 | 描述 |
|-------|---------|------|
| `id` | VARCHAR(256) | 主键，唯一标识符 |
| `doc_id` | VARCHAR | 文档ID |
| `text` | VARCHAR | 文档的自然语言描述，用于 Embedding 和 LLM 上下文 |
| `embedding` | FLOAT_VECTOR | 文本的向量表示，维度由 `settings.MILVUS_EMBEDDING_DIM` 配置 |
| `source_table` | VARCHAR(64) | 数据来源表名 |

### 1.2 实验数据表字段
| 字段名 | 数据类型 | 描述 |
|-------|---------|------|
| `project_library_type` | VARCHAR(256) | 项目建库类型 |
| `species` | VARCHAR(128) | 物种 |
| `sample_type` | VARCHAR(128) | 样本类型 |
| `sample_detailed_type` | VARCHAR(128) | 样本详细类型/组织类型 |
| `tissue` | VARCHAR | 组织 |
| `platform` | VARCHAR | 平台 |
| `category` | VARCHAR | 类别 |
| `prep_method` | VARCHAR(64) | 解离/抽核方法 |
| `arrival_temp_celsius` | FLOAT | 到达温度（摄氏度） |
| `total_cells_10k` | FLOAT | 总细胞数（万） |
| `clumping_rate_percent` | FLOAT | 成团率（百分比） |
| `cell_viability_percent` | FLOAT | 细胞活力（百分比） |
| `nucleated_rate_percent` | FLOAT | 有核率（百分比） |
| `captured_cells` | INT64 | 捕获细胞数 |
| `reads_per_cell` | INT64 | 每个细胞的读取数 |
| `median_genes` | INT64 | 中位数基因数 |
| `annotation_results_full` | VARCHAR(4096) | 完整注释结果 |
| `annotation_cell_types_list` | JSON | 注释的细胞类型列表 |
| `annotation_cell_type_details` | JSON | 注释的细胞类型详情 |
| `tissue_digestion_protocol` | VARCHAR(256) | 组织消化协议 |
| `tissue_digestion_summary` | VARCHAR(4096) | 组织消化摘要 |
| `related_article_link` | VARCHAR(512) | 相关文章链接 |
| `feishu_doc_link` | VARCHAR(512) | 飞书文档链接 |
| `video_link` | VARCHAR(512) | 视频链接 |

### 1.3 样本制备指南表字段
| 字段名 | 数据类型 | 描述 |
|-------|---------|------|
| `product_level1` | VARCHAR(128) | 产品一级分类 |
| `product_level2` | VARCHAR(128) | 产品二级分类 |
| `product_level3` | VARCHAR(128) | 产品三级分类 |
| `sample_category` | VARCHAR(128) | 样本类别 |
| `preparation_method_doc` | VARCHAR(256) | 制备方法文档 |
| `risk_level` | VARCHAR | 风险等级 |
| `recommended_amount_mg` | VARCHAR | 推荐用量（mg） |

## 2. 索引配置

### 2.1 向量索引
- 索引类型：`IVF_FLAT`
- 度量类型：`COSINE`
- 参数：`nlist: 128`（根据数据量优化）

### 2.2 标量字段索引
系统自动为以下重要字段创建 `STL_SORT` 索引，以加速过滤查询：
- `source_table`
- `species`
- `tissue`
- `platform`
- `category`
- `prep_method`
- `risk_level`
- `recommended_amount_mg`

### 2.3 JSON 字段索引
- 字段：`annotation_cell_types_list`
- 索引类型：`JSON`
- 参数：`{"json_path": ".", "index_type": "ARRAY"}`

## 3. 字段映射规则

在 `excel_processor.py` 中定义了中英文列名映射表 `COLUMN_MAPPING`，这是解决 Milvus 报错的关键。所有用于过滤的字段必须在此定义英文名，例如：
- `物种` → `species`
- `组织` → `sample_detailed_type`
- `项目建库类型` → `project_library_type`

## 4. 数据处理流程

1. **数据提取**：从原始文档（Excel）中提取数据
2. **字段映射**：将中文列名映射为标准英文列名
3. **类型转换**：将数值、百分比等转换为适当的数值类型
4. **TextNode 生成**：创建包含文本和元数据的 TextNode
5. **向量化**：生成文本的向量表示
6. **写入 Milvus**：将包含向量的 TextNode 写入 Milvus 集合

## 5. 集合命名规则

- 集合命名格式：`dept_{department}`
- 例如：市场部门的数据存储在 `dept_market` 集合中
- 按部门隔离数据，支持跨部门查询

## 6. 注意事项

1. **字段长度限制**：`VARCHAR` 字段的 `max_length` 需要根据实际数据调整，确保不会截断数据
2. **JSON 字段支持**：Milvus 2.2+ 对 `DataType.JSON` 有很好的支持，可以直接对其内部元素创建索引
3. **重复字段处理**：如果不同表的字段含义相同，可以合并为一个字段；如果含义不同，需要在字段名上做出区分
4. **索引优化**：根据查询模式优化索引配置，提高查询性能
5. **数据隔离**：按部门创建独立集合，确保数据安全和查询效率

## 7. 实际集合示例

根据 `milvus_collections_info_20260119_150448.json` 文件，实际集合包含以下字段：
- `id`
- `doc_id`
- `text`
- `embedding`

在实际使用中，会根据数据类型和业务需求动态添加其他字段。