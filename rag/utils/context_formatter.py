"""
上下文格式化工具 (LLM 轨道 - 零截断版)
职责：移除冗余字段（full_row_json, text, embedding），保留所有业务数据
"""
import json
from typing import List, Dict
from utils.logger import logger


class ContextFormatter:
    """上下文清洗与格式化工具 - 完整保留所有有价值的字段"""
    
    # 需要从 metadata 中剔除的冗余字段（这些字段在其他地方已重复存储）
    EXCLUDE_FIELDS = {
        "full_row_json",        # 重复存储所有字段的JSON字符串 ⚠️ 核心Token杀手
        "text",                 # 自动生成的描述文本（与metadata重复）
        "embedding",            # 向量数据（数千维浮点数,无业务价值）
        "_node_content",        # 内部节点元数据（LlamaIndex内部使用）
        "relationships",        # 节点关系（内部使用）
        "chunk_id",             # 分块ID（内部索引）
        "doc_id",               # 文档ID（已有filename）
        "owner",                # 上传者（与业务无关）
        "uploader",             # 上传者（重复）
        "department",           # 部门（已通过过滤确定）
        "doc_type",             # 文档类型（已知是Excel）
        "row_index",            # 行索引（内部使用）
        "_node_type",           # 节点类型
        "document_id",          # 文档ID
        "ref_doc_id",           # 引用文档ID
        "excluded_embed_metadata_keys",  # 元数据配置
        "excluded_llm_metadata_keys",    # 元数据配置
        "metadata_template",    # 模板配置
        "metadata_separator",   # 分隔符配置
        "metadata_seperator",   # 分隔符配置(typo)
        "text_template",        # 文本模板
        "mimetype",             # MIME类型
        "start_char_idx",       # 字符索引
        "end_char_idx"          # 字符索引
    }
    
    @staticmethod
    def format_for_llm(docs: List, intent: str) -> str:
        """
        格式化为 LLM 上下文（精简版 - 零截断）
        :param docs: 检索到的 Document 列表
        :param intent: 查询意图
        :return: Markdown 格式的精简上下文
        """
        if not docs:
            return "（数据库中未检索到匹配记录）"

        # 提取 metadata
        clean_metas = []
        for doc in docs:
            if hasattr(doc, 'node') and hasattr(doc.node, 'metadata'):
                meta = doc.node.metadata
            elif hasattr(doc, 'metadata'):
                meta = doc.metadata
            elif isinstance(doc, dict):
                meta = doc.get('metadata', {})
            else:
                continue
            clean_metas.append(meta)

        # 根据意图分发
        if intent == "query_experiment_data":
            return ContextFormatter._format_experiment_data(clean_metas)
        elif intent == "query_preparation_guidelines":
            return ContextFormatter._format_preparation_data(clean_metas)
        else:
            return ContextFormatter._format_generic(clean_metas)

    @staticmethod
    def _format_experiment_data(metas: List[Dict]) -> str:
        """
        格式化项目经验数据（完整保留所有有用字段，零截断）
        基于COLUMN_MAPPING中的experiment_data相关字段
        """
        lines = [f"## 检索到 {len(metas)} 条项目经验记录\n"]
        
        for idx, meta in enumerate(metas):
            # 清洗：移除冗余字段
            cleaned_meta = {k: v for k, v in meta.items() 
                           if k not in ContextFormatter.EXCLUDE_FIELDS}
            
            # 辅助函数：安全取值
            def val(k, default="/"): 
                v = cleaned_meta.get(k)
                if v is None or str(v).lower() in ['null', 'none', 'nan', '']:
                    return default
                return str(v)

            # 构建记录（树状结构 - 易于 LLM 解析）
            record = [
                f"### 【记录 {idx+1}】",
                "",
                f"**实验平台**",
                f"- 平台类型: {val('platform_type')}",
                "",
                f"**样本信息**",
                f"- 物种: {val('species')}",
                f"- 样本类型: {val('sample_type_exp')}",
                f"- 样本详细类型: {val('sample_detailed_type')}",
                f"- 实验方案: {val('experiment_protocol')}",
                f"- 组织重量: {val('tissue_weight')} {val('tissue_weight_unit')}",
                f"- 定性描述: {val('qualitative_description')}",
                "",
                f"**样本处理**",
                f"- 是否流式: {val('is_streaming')}",
                f"- 抗体信息: {val('antibody_info')}",
                f"- 流式方案: {val('streaming_protocol')}",
                f"- 是否裂红: {val('is_lysis')}",
                f"- 是否去死: {val('is_dead_removal')}",
                f"- 到样温度: {val('arrival_temp_celsius')}",
                "",
                f"**实验指标**",
                f"- 细胞总量(万): {val('total_cells_10k')}",
                f"- 结团率(%): {val('clumping_rate_percent')}",
                f"- 细胞活率(%): {val('cell_viability_percent')}",
                f"- 有核率(%): {val('nucleated_rate_percent')}",
                f"- RIN值: {val('rin_score')}",
                "",
                f"**数据指标**",
                f"- 捕获细胞数: {val('captured_cells')}",
                f"- Reads/Cell: {val('reads_per_cell')}",
                f"- 基因中位数: {val('median_genes')}",
                "",
                f"**细胞注释结果（完整，零截断）**",  # ⚠️ 重点：完整保留
                f"{val('annotation_results', '未提供注释结果')}",
                "",
                f"**参考资料**",
                f"- 组织消化方案: {val('digestion_protocol')}",
                f"- 相关文章链接: {val('related_articles')}",
                f"- 飞书文档: {val('feishu_doc_link')}",
                f"- 视频直播: {val('video_live_link')}",
                "",
                f"---"
            ]
            
            lines.extend(record)
        
        return "\n".join(lines)

    @staticmethod
    def _format_preparation_data(metas: List[Dict]) -> str:
        """
        格式化样本准备数据（侧重送样量与风险 - 完整保留）
        基于COLUMN_MAPPING中的preparation_guidelines相关字段
        """
        lines = [f"## 检索到 {len(metas)} 条样本准备标准\n"]
        
        for idx, meta in enumerate(metas):
            # 清洗：移除冗余字段
            cleaned_meta = {k: v for k, v in meta.items() 
                           if k not in ContextFormatter.EXCLUDE_FIELDS}
            
            def val(k, default="/"): 
                v = cleaned_meta.get(k)
                if v is None or str(v).lower() in ['null', 'none', '']:
                    return default
                return str(v)

            # 构建记录
            record = [
                f"### 【标准 {idx+1}】",
                "",
                f"**产品信息**",
                f"- 产品一级目录: {val('product_level1')}",
                f"- 产品二级目录: {val('product_level2')}",
                f"- 产品三级目录: {val('product_level3')}",
                "",
                f"**样本制备信息**",
                f"- 样本大类: {val('sample_category_prep')}",
                f"- 样本类型: {val('sample_type_prep')}",
                f"- 组织类型: {val('tissue_type_prep')}",
                f"- 样本处理方式: {val('sample_prep_method')}",
                "",
                f"**送样要求（完整，零截断）**",  # ⚠️ 完整保留
                f"- 建议送样量1: {val('recommended_amount_1')}",
                f"- 建议送样量2: {val('recommended_amount_2')}",
                f"- 建议送样量3: {val('recommended_amount_3')}",
                f"- 定性描述1: {val('qualitative_description_1')}",
                f"- 定性描述2: {val('qualitative_description_2')}",
                f"- 定性描述3: {val('qualitative_description_3')}",
                "",
                f"**质检标准**",
                f"- RIN要求: {val('rin_score')}",
                f"- 细胞活率要求(%): {val('cell_viability_percent')}",
                f"- 细胞总量要求: {val('cell_count')}",
                f"- 活细胞浓度: {val('live_cell_concentration')}",
                f"- 结团率要求: {val('clustering_rate')}",
                f"- 有核率要求(%): {val('nucleated_rate_percent')}",
                "",
                f"**方法和注意事项（完整，零截断）**",  # ⚠️ 完整保留
                f"- 样本准备方法: {val('sample_preparation_method_doc')}",
                f"- 取样送样的注意事项: {val('sampling_notes')}",
                f"- 备注: {val('notes_full_text')}",
                "",
                f"---"
            ]
            
            lines.extend(record)
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_generic(metas: List[Dict]) -> str:
        """
        通用格式化（移除冗余字段，保留所有业务字段）
        """
        lines = [f"## 检索到 {len(metas)} 条记录\n"]
        
        for idx, meta in enumerate(metas):
            # 清洗：移除冗余字段
            cleaned_meta = {k: v for k, v in meta.items() 
                           if k not in ContextFormatter.EXCLUDE_FIELDS}
            
            # 直接转为 JSON 格式（业务字段已足够精简）
            record = f"### 【记录 {idx+1}】\n```json\n{json.dumps(cleaned_meta, ensure_ascii=False, indent=2)}\n```\n"
            lines.append(record)
        
        return "\n".join(lines)