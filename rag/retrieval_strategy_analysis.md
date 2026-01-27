# 检索策略分析：LlamaIndex vs PyMilvus API

## 问题分析

### 1. LlamaIndex检索的核心问题

当前代码中，LlamaIndex检索使用的是 `index.as_retriever()` 创建的检索器，这个检索器默认会执行**语义检索**，即：
1. 先计算查询文本的嵌入向量
2. 在向量空间中进行相似度搜索，获取最相似的结果
3. 然后再应用过滤条件

这就导致了一个关键问题：**语义检索会先过滤掉一些本应符合过滤条件的结果**，因为这些结果可能与查询文本的语义相似度不高，但实际上它们满足了所有的过滤条件。

### 2. PyMilvus API检索的优势

PyMilvus API检索直接构建查询表达式，使用 `collection.query()` 方法进行检索，这种方式：
1. 只基于**精确的过滤条件**进行检索
2. 不进行语义检索，不会因为语义相似度而过滤掉符合条件的结果
3. 可以处理包含特殊字符的字段名（使用反引号包裹）

这就是为什么之前的分析发现，只有使用PyMilvus API检索才能得到正确的结果。

## 代码执行流程

当前结构化表格检索策略的执行流程如下：

1. **构建过滤条件**：包括部门隔离和用户提供的条件
2. **LlamaIndex检索**：
   - 创建检索器，设置 `similarity_top_k=1000`
   - 使用查询文本执行语义检索
   - 应用过滤条件
3. **简化过滤条件重试**：如果结果为空，移除可能有问题的字段，重新执行检索
4. **PyMilvus API降级**：如果LlamaIndex检索失败，使用原始PyMilvus API
5. **简单LlamaIndex检索降级**：如果PyMilvus API也失败，尝试使用不同搜索词

## 代码中存在的bug

在最后一个降级策略中（行545-573），使用了未定义的变量 `main_filters`：

```python
# 只添加不包含特殊字符的过滤条件，提高成功率
for k, v in main_filters:  # 这里 main_filters 未定义，应该是 column_filters
    filters_list.append(MetadataFilter(key=k, operator=FilterOperator.EQ, value=v))
```

这个bug会导致代码执行失败，因为 `main_filters` 变量没有被定义。

## 解决方案建议

### 1. 修复bug

将未定义的 `main_filters` 替换为 `column_filters`，并修正遍历方式：

```python
# 只添加不包含特殊字符的过滤条件，提高成功率
for k, v in column_filters.items():
    if v != "*":
        has_special_chars = any(c in k for c in ['\n', '\r', '\t', '（', '）'])
        if not has_special_chars and 'syfa' not in k.lower():
            filters_list.append(MetadataFilter(key=k, operator=FilterOperator.EQ, value=v))
```

### 2. 优化检索策略

考虑到结构化表格检索的特点，我们更希望基于精确的过滤条件进行检索，而不是基于语义相似度。因此，建议调整检索策略：

1. **优先使用PyMilvus API检索**：对于结构化表格检索，直接使用PyMilvus API进行检索，避免语义检索带来的问题
2. **LlamaIndex检索作为降级方案**：只有在PyMilvus API检索失败时，才使用LlamaIndex检索
3. **调整LlamaIndex检索参数**：如果必须使用LlamaIndex检索，可以考虑：
   - 设置 `similarity_top_k` 为一个很大的值，确保获取所有匹配结果
   - 使用一个中性的搜索词（如" "或"数据"），避免语义检索的影响

### 3. 改进过滤条件处理

统一过滤条件的处理逻辑，确保：
1. 所有过滤条件都能被正确处理
2. 包含特殊字符的字段名能被正确处理
3. 过滤条件的构建过程更加清晰和可靠

## 结论

当前代码中，LlamaIndex检索的语义检索特性确实是导致检索结果不准确的原因。PyMilvus API检索由于只基于精确的过滤条件，因此能得到正确的结果。

此外，代码中还存在一个未定义变量的bug，需要修复。

建议调整检索策略，优先使用PyMilvus API检索，避免语义检索带来的问题，同时修复代码中的bug。