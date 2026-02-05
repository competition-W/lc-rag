"""
数据导出工具 (前端轨道 - 完整数据)
职责：将原始 Milvus 数据转换为前端可展示/下载的完整格式
基于COLUMN_MAPPING的完整字段映射
"""
from typing import List, Dict
from utils.logger import logger


class DataExporter:
    """数据导出工具 - 支持完整字段映射"""
    
    # 字段中文映射表（experiment_data 表 - 基于COLUMN_MAPPING）
    FIELD_MAPPING_EXPERIMENT = {
        # 实验平台
        "platform_type": "实验平台",
        
        # 样本信息
        "species": "物种",
        "sample_type_exp": "样本类型",
        "sample_detailed_type": "样本详细类型",
        "experiment_protocol": "实验方案",
        "tissue_weight": "组织重量",
        "tissue_weight_unit": "组织重量单位",
        "qualitative_description": "定性描述",
        "rin_score": "核酸质量RIN值",
        "is_streaming": "是否流式",
        "antibody_info": "抗体信息",
        "streaming_protocol": "流式方案",
        "is_lysis": "是否裂红",
        "is_dead_removal": "是否去死",
        "arrival_temp_celsius": "到样温度",
        
        # 实验指标
        "total_cells_10k": "细胞总量(万)",
        "clumping_rate_percent": "结团率(%)",
        "cell_viability_percent": "细胞活率(%)",
        "nucleated_rate_percent": "有核率(%)",
        
        # 数据指标
        "captured_cells": "捕获细胞数",
        "reads_per_cell": "reads/cell",
        "median_genes": "基因中位数",
        "annotation_results": "人工细胞注释",
        
        # 参考资料
        "digestion_protocol": "组织消化方案",
        "related_articles": "相关组织用户文章链接",
        "feishu_doc_link": "飞书文档链接",
        "video_live_link": "视频直播链接",
        
        # 元数据
        "filename": "来源文件",
        "sheet_name": "表单名称"
    }
    
    # 字段中文映射表（preparation_guidelines 表 - 基于COLUMN_MAPPING）
    FIELD_MAPPING_PREPARATION = {
        # 产品信息
        "product_level1": "产品一级目录",
        "product_level2": "产品二级目录",
        "product_level3": "产品三级目录",
        
        # 样本制备信息
        "sample_category_prep": "样本大类",
        "sample_type_prep": "样本类型",
        "tissue_type_prep": "组织类型",
        "sample_prep_method": "样本处理方式",
        
        # 送样要求
        "recommended_amount_1": "建议送样量1",
        "recommended_amount_2": "建议送样量2",
        "recommended_amount_3": "建议送样量3",
        "qualitative_description_1": "定性描述1",
        "qualitative_description_2": "定性描述2",
        "qualitative_description_3": "定性描述3",
        
        # 质检标准
        "rin_score": "质检标准_RIN",
        "cell_viability_percent": "质检标准_细胞活率",
        "cell_count": "质检标准_细胞总量",
        "live_cell_concentration": "质检标准_活细胞浓度",
        "clustering_rate": "质检标准_结团率",
        "nucleated_rate_percent": "质检标准_有核率",
        
        # 方法和注意事项
        "sample_preparation_method_doc": "样本准备方法",
        "sampling_notes": "取样送样的注意事项",
        "notes_full_text": "备注",
        
        # 元数据
        "filename": "来源文件",
        "sheet_name": "表单名称"
    }
    
    @staticmethod
    def export_for_frontend(docs: List, intent: str) -> List[Dict]:
        """
        导出完整数据供前端使用（包含所有字段，用于Excel下载）
        :param docs: 原始 Document 列表
        :param intent: 查询意图
        :return: 字典列表，包含所有原始字段
        """
        result = []
        
        # 根据意图选择字段映射
        if intent == "query_experiment_data":
            field_map = DataExporter.FIELD_MAPPING_EXPERIMENT
        elif intent == "query_preparation_guidelines":
            field_map = DataExporter.FIELD_MAPPING_PREPARATION
        else:
            # 未知意图，尝试合并两个映射表
            field_map = {**DataExporter.FIELD_MAPPING_EXPERIMENT, 
                        **DataExporter.FIELD_MAPPING_PREPARATION}
        
        for doc in docs:
            # 提取 metadata
            if hasattr(doc, 'node') and hasattr(doc.node, 'metadata'):
                meta = doc.node.metadata
            elif hasattr(doc, 'metadata'):
                meta = doc.metadata
            elif isinstance(doc, dict):
                meta = doc.get('metadata', {})
            else:
                continue
            
            # 构建前端数据行（保留所有有用字段）
            row = {}
            for en_key, cn_key in field_map.items():
                value = meta.get(en_key, "")
                # 保留空值，让前端决定如何展示
                row[cn_key] = value
            
            # 保留原始 ID 用于溯源
            row["_id"] = meta.get("id", "")
            row["_source_table"] = meta.get("source_table", "")
            
            result.append(row)
        
        logger.info(f"✅ DataExporter: 导出 {len(result)} 条完整记录供前端下载")
        
        return result