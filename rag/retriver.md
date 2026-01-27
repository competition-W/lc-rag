太好了！数据预处理是整个流程的基石。既然你已经按照方案对数据进行了扁平化处理，并将其统一为Milvus可摄入的格式（每个文档包含 `id`, `text`, `metadata`，且`metadata`中的字段都已标准化），我们就可以构建后续的RAG检索与生成流程了。

下面是基于你已处理好的数据和Milvus实现的完整方案：

### **Milvus 架构与数据摄入**

1.  **Milvus Collection Schema:**
    *   根据你预处理后的数据，确保在Milvus中创建的Collection Schema包含了所有你需要查询和显示的`metadata`字段。
    *   **关键点：** 对所有你计划进行过滤的`metadata`字段（例如：`species`, `sample_detailed_type`, `prep_method`, `risk_level`, `recommended_amount_mg`, `annotation_cell_types_list` 等），都**创建索引**。对于JSON类型的字段（如`annotation_cell_types_list`），Milvus支持对其内部元素进行索引。
    *   **示例 Schema (假设你已经创建并加载):**
        ```python
        # from pymilvus import FieldSchema, CollectionSchema, DataType, Collection, utility

        # # 确保所有需要过滤的字段都有索引
        # utility.index_building_progress("your_collection_name") 
        # collection = Collection("your_collection_name")
        # collection.load() # 确保 collection 加载到内存
        ```
2.  **数据摄入：**
    *   使用你预处理好的JSONL文件，对每个文档生成 `text` 字段的嵌入（Embedding）。
    *   将 `id`、`embedding` 和 `metadata` 中的所有字段（作为独立的参数）插入到Milvus Collection中。
    *   **注意：** 不要将整个`metadata`字典作为一个JSON字符串插入到Milvus的某个字段中，而是应该将`metadata`中的每个键值对作为Collection schema中的独立字段插入。这是因为Milvus的过滤主要针对这些独立字段。

---

### **RAG 整体流程：三阶段实现**

### **阶段一：增强型意图识别与实体抽取 (LLM-based)**

这是整个RAG流程的入口，负责将用户的自然语言查询转化为结构化的查询条件。

1.  **输入：** 用户的自然语言查询（`original_user_query`）。
2.  **工具：** 大语言模型 (LLM)，例如 OpenAI GPT 系列、Claude、Llama 等。
3.  **Prompt 设计：**
    *   需要明确告知LLM可以识别哪些字段（这些字段名应与你预处理后的`metadata`字段名一致）。
    *   指导LLM以JSON格式输出，并定义好JSON的结构。
    *   需要能够区分查询意图（实验数据 vs 制备指南），以便后续的过滤和结果格式化。

    ```python
    def get_intent_and_filters(user_query):
        """
        使用LLM解析用户查询，提取意图、Milvus过滤器和请求字段。
        """
        # 注意：这里需要替换为你实际使用的LLM API调用
        # 示例Prompt，需要根据实际LLM和具体字段进行调整和优化
        prompt = f"""
        你是一个能够精确分析用户查询的助手。请从以下用户问题中提取查询意图、用于Milvus精确过滤的条件，以及用户最终想要获取的具体信息字段。

        请严格按照以下JSON格式输出。如果某个字段在查询中未明确提及，请将其值设为 null 或空列表（根据字段类型）。

        可识别的意图：
        - "query_experiment_data": 查询实验记录相关信息。
        - "query_preparation_guidelines": 查询样本制备指南相关信息。
        - "general_query": 如果无法明确区分上述两种意图。

        可用于过滤的实验数据字段 (来自 'experiment_data' 源):
        - `source_table`: "experiment_data" (如果明确指代实验数据)
        - `project_library_type` (e.g., "10X单细胞3‘转录本-抽核(V3试剂)")
        - `species` (e.g., "小鼠", "人")
        - `sample_type` (e.g., "冻存组织", "新鲜组织")
        - `sample_detailed_type` (e.g., "心肌组织", "肿瘤")
        - `prep_method` (e.g., "解离", "抽核")
        - `arrival_temp_celsius` (数值, e.g., "-80")
        - `total_cells_10k` (数值, 细胞总量(万))
        - `clumping_rate_percent` (数值, 结团率(%))
        - `cell_viability_percent` (数值, 细胞活率(%))
        - `nucleated_rate_percent` (数值, 有核率(%))
        - `captured_cells` (数值, 捕获细胞数)
        - `reads_per_cell` (数值, reads/cell)
        - `median_genes` (数值, 基因中位数)
        - `target_cell_types`: (字符串列表, e.g., ["巨噬细胞", "内皮细胞"], 如果用户询问“有没有鉴定到巨噬细胞？”则提取“巨噬细胞”)
        - `tissue_digestion_protocol` (e.g., "温和机械解离")
        - `tissue_digestion_summary` (e.g., "使用剪刀剪碎后，温和吹打")
        - `related_article_link` (文章链接)
        - `feishu_doc_link` (飞书文档链接)
        - `video_link` (视频链接)

        可用于过滤的样本制备指南字段 (来自 'preparation_guidelines' 源):
        - `source_table`: "preparation_guidelines" (如果明确指代制备指南)
        - `product_level1` (e.g., "单细胞空间组学")
        - `product_level2` (e.g., "单细胞测序")
        - `product_level3` (e.g., "单细胞转录组")
        - `sample_category` (e.g., "实体组织类样本")
        - `sample_type` (e.g., "新鲜实体组织")
        - `tissue_type` (e.g., "肿瘤", "脑组织")
        - `prep_method` (e.g., "解离", "抽核")
        - `preparation_method_doc` (SOP文档名)
        - `handling_notes` (注意事项)
        - `risk_level` (e.g., "风险", "合格")
        - `recommended_amount_mg` (数值, 建议送样量(mg))
        - `qualitative_description_text` (e.g., "大米大小", "绿豆大小")
        - `risk_query`: (布尔值, 如果用户询问“有没有风险？”则设置为 true)

        用户明确想获取的信息字段 (请使用预处理后的字段名):
        - `requested_fields`: 字符串列表 (e.g., ["cell_viability_percent", "annotation_results_full", "notes_full"])

        用户问题: "{user_query}"

        输出JSON格式:
        ```json
        {{
          "query_intent": "query_experiment_data", // 或 "query_preparation_guidelines", "general_query"
          "milvus_filters": {{
            "source_table": null, 
            "species": null,
            "sample_detailed_type": null,
            "prep_method": null,
            "cell_viability_percent": null, // 可以是具体值或范围 {'op': 'gt', 'value': 90}
            "target_cell_types": [], // 例如，用户问“有没有鉴定到巨噬细胞”，则为 ["巨噬细胞"]
            "risk_query": false, // 如果用户问“有没有风险”
            "risk_level": null, // 如果用户问“风险的样本”，则为 "风险"
            "recommended_amount_mg": null, // 如果用户问“20mg的样本”，则为 20.0
            // ... 列出所有可能的Milvus过滤字段
          }},
          "requested_fields": [] // e.g., ["annotation_results_full", "notes_full", "handling_notes"]
        }}
        ```
        请注意：对于数值字段，如果用户查询是范围（例如“活率大于90%”），LLM应该将其解析为 `{"cell_viability_percent": {"op": "gt", "value": 90}}` 这样的结构。
        """
        
        # 实际LLM调用，这里只是一个占位符
        llm_response_str = call_your_llm_api(prompt, temperature=0) # temperature=0 追求确定性
        
        try:
            intent_recognition_result = json.loads(llm_response_str)
            # 兼容处理数值范围查询，以便后续构建Milvus expr
            for k, v in intent_recognition_result["milvus_filters"].items():
                if isinstance(v, dict) and 'op' in v and 'value' in v:
                    pass # 已经是期望的格式
                elif isinstance(v, str) and (v.startswith('>') or v.startswith('<') or v.startswith('=') or v.startswith('!=')):
                    # 如果LLM直接返回字符串如 ">90"，需要解析
                    match = re.match(r'([<>=!]+)\s*(\d+\.?\d*)', v)
                    if match:
                        op = match.group(1)
                        val = float(match.group(2))
                        intent_recognition_result["milvus_filters"][k] = {"op": op, "value": val}
            return intent_recognition_result
        except json.JSONDecodeError as e:
            print(f"LLM响应解析失败: {e}\n原始响应: {llm_response_str}")
            # 错误处理：返回一个默认的通用查询结构
            return {
                "query_intent": "general_query",
                "milvus_filters": {},
                "requested_fields": []
            }

    # 示例调用
    # original_user_query = "小鼠心肌组织抽核有没有鉴定到巨噬细胞？细胞活率怎么样？"
    # parsed_query = get_intent_and_filters(original_user_query)
    # print(json.dumps(parsed_query, indent=2, ensure_ascii=False))
    ```

### **阶段二：Milvus 精确检索**

利用阶段一解析出的结构化过滤器，从Milvus中检索所有匹配的文档。

1.  **输入：** `parsed_query` (包含 `milvus_filters`, `query_intent` 等)。
2.  **工具：** `pymilvus` 客户端。
3.  **构建 Milvus 查询表达式 (DSL)：**
    *   根据 `milvus_filters` 动态构建 `expr` 字符串。
    *   处理数值范围查询（例如 `cell_viability_percent > 90`）。
    *   处理`annotation_cell_types_list`的`array_contains`操作。
    *   根据`query_intent`，可以预先添加`source_table`过滤条件。

    ```python
    def build_milvus_expr(filters):
        """根据解析出的过滤器构建Milvus的查询表达式。"""
        conditions = []
        for field, value in filters.items():
            if value is None or (isinstance(value, (list, dict)) and not value): # 忽略空值或空列表/字典
                continue

            # 处理数值范围查询，例如 {"op": "gt", "value": 90}
            if isinstance(value, dict) and 'op' in value and 'value' in value:
                op_map = {
                    "gt": ">", "lt": "<", "eq": "==", "ne": "!=",
                    "gte": ">=", "lte": "<="
                }
                op = op_map.get(value['op'], "==")
                conditions.append(f"{field} {op} {value['value']}")
            # 处理目标细胞类型 (JSON array_contains)
            elif field == "target_cell_types" and isinstance(value, list) and value:
                for cell_type in value:
                    conditions.append(f"annotation_cell_types_list array_contains '{cell_type}'")
            # 处理一般字符串/数值精确匹配
            elif isinstance(value, str):
                conditions.append(f"{field} == '{value}'")
            elif isinstance(value, (int, float, bool)):
                conditions.append(f"{field} == {value}")
            # ... 其他复杂类型的处理

        if not conditions:
            return "" # 无过滤条件
        return " && ".join(conditions)

    def retrieve_from_milvus(parsed_query, collection):
        """
        从Milvus中检索匹配的文档。
        """
        milvus_filters = parsed_query["milvus_filters"]
        
        # 根据意图添加 source_table 过滤
        if parsed_query["query_intent"] == "query_experiment_data":
            milvus_filters["source_table"] = "experiment_data"
        elif parsed_query["query_intent"] == "query_preparation_guidelines":
            milvus_filters["source_table"] = "preparation_guidelines"
        # 如果是 general_query，则不限制 source_table

        expr = build_milvus_expr(milvus_filters)
        print(f"Milvus 查询表达式: {expr}")

        # 请求所有可能的字段，以便后续灵活使用
        all_possible_fields = list(collection.schema.fields_by_name.keys())
        # 排除 embedding 字段
        output_fields = [f.name for f in collection.schema.fields if f.name != "embedding"]

        try:
            hits = collection.query(
                expr=expr,
                output_fields=output_fields,
                limit=100000,  # 设置一个足够大的限制，以确保返回所有匹配的记录
                consistency_level="Strong" # 保证查询结果的强一致性
            )
            retrieved_documents = [hit for hit in hits] # Milvus的query方法直接返回字段值
            return retrieved_documents
        except Exception as e:
            print(f"Milvus query failed: {e}")
            return []

    # 示例调用
    # collection = Collection("your_collection_name") # 假设 collection 已初始化并加载
    # retrieved_docs = retrieve_from_milvus(parsed_query, collection)
    # print(f"检索到 {len(retrieved_docs)} 条记录。")
    ```

### **阶段三：上下文构建与LLM响应生成**

这是处理大量检索结果并生成用户友好回答的核心阶段。

1.  **输入：** `original_user_query`, `parsed_query`, `retrieved_documents`。
2.  **工具：** 大语言模型 (LLM)，`tiktoken` (用于准确的token估算)。

    ```python
    import tiktoken # pip install tiktoken

    def format_single_document(doc, parsed_query):
        """
        根据文档内容和用户请求，格式化单条文档的显示。
        这里需要根据 'source_table' 和 'requested_fields' 以及特定查询逻辑来定制。
        """
        doc_info = []
        doc_id = doc.get("id", "N/A")
        doc_info.append(f"--- 记录 ID: {doc_id} ---")
        
        source_table = doc.get("source_table")
        requested_fields = parsed_query.get("requested_fields", [])
        milvus_filters = parsed_query.get("milvus_filters", {})

        if source_table == "experiment_data":
            doc_info.append(f"来源: 实验数据")
            doc_info.append(f"项目建库类型: {doc.get('project_library_type', 'N/A')}")
            doc_info.append(f"物种: {doc.get('species', 'N/A')}")
            doc_info.append(f"样本详细类型: {doc.get('sample_detailed_type', 'N/A')}")
            doc_info.append(f"实验方案: {doc.get('prep_method', 'N/A')}")

            # 处理特定请求字段
            for field in requested_fields:
                if field == "annotation_results_full":
                    # 如果用户询问特定细胞类型，尝试从 annotation_cell_type_details 中提取
                    target_cells = milvus_filters.get('target_cell_types')
                    if target_cells:
                        details = doc.get('annotation_cell_type_details', {})
                        found_cells_info = []
                        for cell_type in target_cells:
                            if cell_type in details:
                                found_cells_info.append(f"细胞类型 '{cell_type}' 鉴定到，占比: {details[cell_type]}%")
                            else:
                                found_cells_info.append(f"细胞类型 '{cell_type}' 未鉴定到。")
                        if found_cells_info:
                            doc_info.append(" | ".join(found_cells_info))
                        else: # 如果没匹配到但仍要求完整注释，则显示
                             doc_info.append(f"人工注释结果: {doc.get('annotation_results_full', 'N/A')}")
                    else: # 如果没有指定细胞类型，直接显示完整注释
                        doc_info.append(f"人工注释结果: {doc.get('annotation_results_full', 'N/A')}")
                elif field == "cell_viability_percent":
                    doc_info.append(f"细胞活率: {doc.get('cell_viability_percent', 'N/A')}%")
                elif field == "related_article_link":
                    doc_info.append(f"相关文章链接: {doc.get('related_article_link', 'N/A')}")
                # ... 其他 requested_fields 的处理
                else:
                    value = doc.get(field, 'N/A')
                    doc_info.append(f"{field.replace('_', ' ').title()}: {value}") # 简单格式化字段名

            # 如果没有requested_fields，可以添加一些默认的核心指标
            if not requested_fields:
                doc_info.append(f"细胞活率: {doc.get('cell_viability_percent', 'N/A')}%")
                doc_info.append(f"捕获细胞数: {doc.get('captured_cells', 'N/A')}")
                doc_info.append(f"人工注释结果 (部分): {doc.get('annotation_results_full', 'N/A')[:100]}...") # 避免过长

        elif source_table == "preparation_guidelines":
            doc_info.append(f"来源: 样本制备指南")
            doc_info.append(f"产品类型: {doc.get('product_level1', 'N/A')} > {doc.get('product_level2', 'N/A')} > {doc.get('product_level3', 'N/A')}")
            doc_info.append(f"样本大类: {doc.get('sample_category', 'N/A')}")
            doc_info.append(f"样本类型: {doc.get('sample_type', 'N/A')}")
            doc_info.append(f"组织类型: {doc.get('tissue_type', 'N/A')}")
            doc_info.append(f"制备方案: {doc.get('prep_method', 'N/A')}")
            doc_info.append(f"建议送样量: {doc.get('recommended_amount_mg', 'N/A')}mg (约{doc.get('qualitative_description_text', 'N/A')})")
            doc_info.append(f"风险级别: {doc.get('risk_level', 'N/A')}")

            # 处理特定请求字段
            for field in requested_fields:
                if field == "notes_full":
                    # 如果用户询问风险，突出显示
                    if milvus_filters.get('risk_query'):
                        doc_info.append(f"备注 (风险说明): {doc.get('notes_full', 'N/A')}")
                    else:
                        doc_info.append(f"备注: {doc.get('notes_full', 'N/A')}")
                elif field == "handling_notes":
                    doc_info.append(f"取样送样的注意事项: {doc.get('handling_notes', 'N/A')}")
                elif field == "preparation_method_doc":
                    doc_info.append(f"样本准备方法SOP: {doc.get('preparation_method_doc', 'N/A')}")
                # ... 其他 requested_fields 的处理
                else:
                    value = doc.get(field, 'N/A')
                    doc_info.append(f"{field.replace('_', ' ').title()}: {value}")

            # 如果没有requested_fields，可以添加一些默认的核心信息
            if not requested_fields:
                doc_info.append(f"风险级别: {doc.get('risk_level', 'N/A')}")
                doc_info.append(f"备注 (部分): {doc.get('notes_full', 'N/A')[:100]}...")


        return "\n".join(doc_info)

    def get_token_count(text, model_name="gpt-4"):
        """使用tiktoken估算文本的token数量"""
        encoding = tiktoken.encoding_for_model(model_name)
        return len(encoding.encode(text))

    def generate_llm_response(original_user_query, parsed_query, retrieved_documents, max_context_tokens=100000):
        """
        构建LLM上下文并生成最终回答。
        """
        if not retrieved_documents:
            return "抱歉，根据您的条件，我没有找到任何相关信息。请尝试修改您的查询。"

        full_formatted_context_parts = []
        for doc in retrieved_documents:
            full_formatted_context_parts.append(format_single_document(doc, parsed_query))
        
        full_formatted_context = "\n\n".join(full_formatted_context_parts)
        
        # --- 上下文长度管理策略 ---
        final_llm_context = ""
        user_guidance_message = ""
        
        # 估计Prompt中其他部分的token数量，给实际数据留出空间
        # 这是一个粗略估计，需要根据最终Prompt结构和LLM模型调整
        overhead_tokens = get_token_count(original_user_query) + 500 # 留一些Buffer给指令和LLM生成答案
        available_context_tokens = max_context_tokens - overhead_tokens

        # 如果所有数据能装下
        if get_token_count(full_formatted_context) <= available_context_tokens:
            final_llm_context = full_formatted_context
            user_guidance_message = f"以下是您查询到的全部 {len(retrieved_documents)} 条符合条件的记录的详细信息："
        else:
            # 策略：展示前N条详细信息 + 剩余数据的概览 + 下载链接
            num_docs_to_show_detail = 5 # 默认显示前5条详细记录
            detailed_part_tokens = 0
            detailed_docs_count = 0
            detailed_context_parts = []

            # 动态决定显示多少条详细记录，确保不超过上下文限制
            for doc_part in full_formatted_context_parts:
                if detailed_part_tokens + get_token_count(doc_part) < available_context_tokens * 0.5: # 详细部分最多占一半上下文
                    detailed_context_parts.append(doc_part)
                    detailed_part_tokens += get_token_count(doc_part)
                    detailed_docs_count += 1
                else:
                    break
            
            detail_context = "\n\n".join(detailed_context_parts)
            remaining_docs = retrieved_documents[detailed_docs_count:]

            summary_of_remaining = []
            if remaining_docs:
                summary_of_remaining.append(f"\n\n--- 剩余 {len(remaining_docs)} 条记录概览 ---")
                
                # 对剩余文档进行统计性概括
                # 收集所有相关数值字段的值
                numeric_fields = [
                    'total_cells_10k', 'clumping_rate_percent', 'cell_viability_percent', 
                    'nucleated_rate_percent', 'captured_cells', 'reads_per_cell', 'median_genes',
                    'recommended_amount_mg'
                ]
                for field in numeric_fields:
                    values = [doc.get(field) for doc in remaining_docs 
                              if doc.get(field) is not None and isinstance(doc.get(field), (int, float))]
                    if values:
                        avg_val = sum(values) / len(values)
                        min_val = min(values)
                        max_val = max(values)
                        summary_of_remaining.append(f"{field.replace('_', ' ').title()}: 平均值 {avg_val:.2f}, 范围 [{min_val:.2f}, {max_val:.2f}]")
                
                # 对于分类字段，可以列出最常见的几个
                categorical_fields = ['species', 'sample_detailed_type', 'risk_level', 'prep_method']
                for field in categorical_fields:
                    values = [doc.get(field) for doc in remaining_docs if doc.get(field) is not None]
                    if values:
                        from collections import Counter
                        counts = Counter(values)
                        most_common = counts.most_common(3) # 最常见的3个
                        summary_of_remaining.append(f"{field.replace('_', ' ').title()}: 主要包括 {', '.join([f'{k} ({v}条)' for k, v in most_common])} 等。")

                summary_of_remaining.append("更多详细信息请参考完整数据导出或尝试更精确的查询。")
            
            summary_context = "\n".join(summary_of_remaining)
            
            # 再次检查，确保详细部分和概览部分加起来不超过上下文
            # 如果概览部分太长，也要截断
            if get_token_count(detail_context + "\n" + summary_context) > available_context_tokens:
                # 进一步截断概览部分，或只显示一部分详细信息
                # 这里的逻辑可以更复杂，例如尝试用LLM对summary_context进行摘要
                final_llm_context = detail_context + "\n\n" + summary_context[:int(available_context_tokens - get_token_count(detail_context)) * 2 // 3] # 粗略截断
                if get_token_count(summary_context) > (available_context_tokens - get_token_count(detail_context)):
                    final_llm_context += "...\n(概览部分过长，已截断)"
            else:
                 final_llm_context = detail_context + "\n\n" + summary_context

            user_guidance_message = (
                f"检测到 {len(retrieved_documents)} 条符合条件的记录。由于数据量庞大，"
                f"以下为您展示了前 {detailed_docs_count} 条记录的详细信息，并对剩余 {len(remaining_docs)} 条记录进行了概括。 "
                f"如需查看所有详细数据，请使用下载链接或进一步细化您的查询。"
            )
            # Placeholder for actual download link generation
            # full_data_download_link = generate_download_link(retrieved_documents) 
            # user_guidance_message += f"\n[点击此处下载完整数据]({full_data_download_link})"

        # --- 构建最终发送给LLM的Prompt ---
        final_prompt_to_llm = f"""
        你是一个专业的生物信息分析助手。请根据以下用户问题和提供的“相关数据”来生成一个全面且准确的回答。

        重要指示：
        1.  请首先向用户说明数据量的处理情况：是展示了所有数据，还是部分数据+概括。这部分内容已经在“用户指导信息”中提供，请作为回答的开场白。
        2.  严格依据“相关数据”回答用户的问题，不要臆造、猜测或补充额外信息。
        3.  如果“相关数据”中包含了用户询问的特定细胞类型，请明确指出其占比。
        4.  回答时请条理清晰，如果有多条记录，请逐一说明；如果是概括信息，请提供关键统计数据。
        5.  如果“用户指导信息”中提到下载链接，请在回答的末尾加上。

        用户问题: "{original_user_query}"

        用户指导信息:
        {user_guidance_message}

        相关数据:
        {final_llm_context}
        ---

        请开始你的回答：
        """
        
        # 实际LLM调用
        # llm_final_response = call_your_llm_api(final_prompt_to_llm)
        # return llm_final_response
        return f"--- 最终发送给LLM的Prompt ---\n{final_prompt_to_llm}" # 仅为演示，实际应调用LLM
    ```

### **完整端到端流程示例**

```python
# 假设 Milvus collection 已经初始化并加载
# from pymilvus import Collection
# collection = Collection("your_collection_name") 
# collection.load() 

# 1. 用户输入
original_user_query = "小鼠心肌组织抽核有没有鉴定到巨噬细胞？细胞活率怎么样？"

# 2. 意图识别与实体抽取
parsed_query = get_intent_and_filters(original_user_query)
print(f"\n--- 阶段一：解析用户查询 ---\n{json.dumps(parsed_query, indent=2, ensure_ascii=False)}")

# 3. Milvus 精确检索
# retrieved_docs = retrieve_from_milvus(parsed_query, collection) # 实际调用
# 模拟 Milvus 返回的文档，确保包含所有可能的字段
mock_doc1 = {
    "id": "exp_001",
    "source_table": "experiment_data",
    "project_library_type": "10X单细胞3‘转录本-抽核(V3试剂)",
    "species": "小鼠",
    "sample_type": "冻存组织",
    "sample_detailed_type": "心肌组织",
    "prep_method": "抽核",
    "arrival_temp_celsius": -80.0,
    "total_cells_10k": 10.0,
    "clumping_rate_percent": 1.2,
    "cell_viability_percent": 95.0, # 高活率
    "nucleated_rate_percent": 99.0,
    "captured_cells": 14466,
    "reads_per_cell": 22517,
    "median_genes": 1259,
    "annotation_results_full": "Endothelial(38.6%),Fibroblast(26.34%),Adipocyte(12.52%),Macrophage(10.09%),Pericyte(9.1%)",
    "annotation_cell_types_list": ["Endothelial", "Fibroblast", "Adipocyte", "Macrophage", "Pericyte"],
    "annotation_cell_type_details": {"Endothelial": 38.6, "Fibroblast": 26.34, "Adipocyte": 12.52, "Macrophage": 10.09, "Pericyte": 9.1},
    "tissue_digestion_protocol": "温和机械解离",
    "tissue_digestion_summary": "使用剪刀剪碎后，温和吹打",
    "related_article_link": "http://example.com/article1",
    "feishu_doc_link": "http://example.com/feishu1",
    "video_link": "http://example.com/video1"
}
mock_doc2 = {
    "id": "exp_002",
    "source_table": "experiment_data",
    "project_library_type": "10X单细胞3‘转录本-抽核(V3试剂)",
    "species": "小鼠",
    "sample_type": "冻存组织",
    "sample_detailed_type": "心肌组织",
    "prep_method": "抽核",
    "arrival_temp_celsius": -80.0,
    "total_cells_10k": 8.0,
    "clumping_rate_percent": 2.5,
    "cell_viability_percent": 88.5, # 中等活率
    "nucleated_rate_percent": 98.0,
    "captured_cells": 12000,
    "reads_per_cell": 20000,
    "median_genes": 1100,
    "annotation_results_full": "Endothelial(45%),Fibroblast(30%),T cell(15%),Macrophage(5%)",
    "annotation_cell_types_list": ["Endothelial", "Fibroblast", "T cell", "Macrophage"],
    "annotation_cell_type_details": {"Endothelial": 45.0, "Fibroblast": 30.0, "T cell": 15.0, "Macrophage": 5.0},
    "tissue_digestion_protocol": "温和机械解离",
    "tissue_digestion_summary": "使用剪刀剪碎后，温和吹打",
    "related_article_link": "http://example.com/article2",
    "feishu_doc_link": "http://example.com/feishu2",
    "video_link": "http://example.com/video2"
}
# 假设有几百条类似的数据
retrieved_docs = [mock_doc1, mock_doc2] * 50 # 模拟100条记录

print(f"\n--- 阶段二：Milvus 检索到 {len(retrieved_docs)} 条记录 ---")

# 4. 上下文构建与LLM响应生成
llm_final_response = generate_llm_response(original_user_query, parsed_query, retrieved_docs, max_context_tokens=8000) # 假设LLM上下文8k
print(f"\n--- 阶段三：LLM最终响应 ---")
print(llm_final_response)

# 测试查询样本制备指南
# original_user_query_prep = "肿瘤组织解离送样量20mg有没有风险？注意事项是什么？"
# parsed_query_prep = get_intent_and_filters(original_user_query_prep)
# print(f"\n--- 阶段一：解析用户查询 (制备指南) ---\n{json.dumps(parsed_query_prep, indent=2, ensure_ascii=False)}")

# mock_prep_doc = {
#     "id": "prep_001_risk_20mg",
#     "source_table": "preparation_guidelines",
#     "product_level1": "单细胞空间组学",
#     "product_level2": "单细胞测序",
#     "product_level3": "单细胞转录组",
#     "sample_category": "实体组织类样本",
#     "sample_type": "新鲜实体组织",
#     "tissue_type": "肿瘤",
#     "prep_method": "解离",
#     "preparation_method_doc": "实体组织送样方法sop",
#     "handling_notes": "避免重复冻融；尽快送样",
#     "risk_level": "风险",
#     "recommended_amount_mg": 20.0,
#     "qualitative_description_text": "大米大小",
#     "notes_full": "风险：可以尝试，细胞量接近上机捕获极限，坏死类组织需根据坏死部位占比来提高送样量；"
# }
# retrieved_prep_docs = [mock_prep_doc]
# print(f"\n--- 阶段二：Milvus 检索到 {len(retrieved_prep_docs)} 条记录 (制备指南) ---")

# llm_final_response_prep = generate_llm_response(original_user_query_prep, parsed_query_prep, retrieved_prep_docs, max_context_tokens=8000)
# print(f"\n--- 阶段三：LLM最终响应 (制备指南) ---")
# print(llm_final_response_prep)
```

### **关键考量与优化**

1.  **LLM选择与API集成：** 替换 `call_your_llm_api` 为你实际使用的LLM服务（如OpenAI SDK、Hugging Face Transformers等）。
2.  **`tiktoken` 模型名：** 估算token时，确保 `get_token_count` 中使用的 `model_name` 与你实际使用的LLM模型相匹配。
3.  **`max_context_tokens`：** 准确设置你的LLM支持的最大上下文窗口大小。
4.  **`format_single_document` 的定制化：** 这是你根据用户需求定制输出内容的关键函数。务必仔细设计，确保它能清晰、准确地呈现用户想看的信息。你可以为不同`source_table`或`query_intent`提供不同的格式化逻辑。
5.  **下载链接：** `generate_download_link(retrieved_documents)` 是一个待实现的函数。它需要将`retrieved_documents`保存到一个临时文件或云存储，并返回一个可访问的URL。
6.  **错误处理：** 在实际部署中，需要更健壮的错误处理和日志记录机制。
7.  **性能：** 对于极大规模的检索结果（数千甚至上万条），即便做了概括，生成概括本身也可能耗时。此时可能需要：
    *   进一步细化用户的查询，鼓励用户添加更多过滤条件。
    *   引入更高效的聚合统计服务。
    *   考虑异步处理或流式传输结果。
8.  **缓存：** 对于频繁查询且结果变化不大的情况，可以考虑缓存Milvus的检索结果。
9.  **Prompt Engineering 迭代：** LLM的性能高度依赖于Prompt。随着你对用户查询模式和数据特点的理解加深，不断优化 `get_intent_and_filters` 和 `generate_llm_response` 中的Prompt是至关重要的。

这个方案为你提供了一个完整且可操作的框架，可以在处理大规模结构化数据时兼顾精确检索和LLM上下文管理。
