# 文件: /mnt/omicshub/rag/tasks/document_tasks_async.py

"""
处理文档的异步任务，包含主要的表格处理逻辑
"""
import asyncio
import re
import uuid
import json
from typing import Dict, Any, List, Tuple
from celery import Task
from llama_index.core.schema import TextNode
from llama_index.core import Settings
from llm.async_embedding import get_async_embed_model
from .celery_app import celery_app
from services.minio_client import minio_client
from services.processors.factory import DocumentProcessorFactory
from services.milvus_manager import milvus_manager
from utils.logger import logger

# ==========================================
# 辅助函数定义
# ==========================================

def clean_and_convert_value(value, target_type=str):
    """
    通用清洗和类型转换函数
    
    Args:
        value: 原始值
        target_type: 目标类型 (str, int, float, bool)
        
    Returns:
        转换后的值，无法转换则返回None
    """
    # 处理None值
    if value is None:
        return None
    
    # 处理pandas Series类型，取第一个有效值
    if hasattr(value, 'tolist'):
        try:
            # 转换为列表并过滤掉None和空字符串
            value_list = [v for v in value.tolist() if v is not None and v != ""]
            if value_list:
                # 递归调用处理第一个有效值
                return clean_and_convert_value(value_list[0], target_type)
            else:
                return None
        except:
            return None
    
    # 处理pandas的NA/NAN类型
    elif hasattr(value, '__class__') and value.__class__.__name__ in ['NA', 'NAType']:
        return None
    
    # 处理浮点数NaN
    elif isinstance(value, float) and value != value:
        return None
    
    # 处理空字符串
    elif isinstance(value, str) and value == "":
        return None
    
    try:
        original_value = value  # 保留原始值，用于特殊情况处理
        
        if isinstance(value, str):
            # 去除首尾空格
            value = value.strip()
            
            # 处理特殊值 "/"和"(无)"，直接返回 None
            if value == "/" or value == "(无)":
                return None
            
            # 处理布尔值
            if target_type == bool:
                if value.lower() in ["是", "yes", "true", "1", "y"]:
                    return True
                elif value.lower() in ["否", "no", "false", "0", "n"]:
                    return False
            
            # 根据目标类型决定是否移除逗号和百分号
            if target_type in [int, float]:
                value = value.replace(",", "").strip()  # 数值类型移除逗号
                value = value.replace("%", "").strip()  # 数值类型移除百分号
                
                # 移除百分号后如果为空，返回 None
                if value == "":
                    return None
        
        if target_type == int:
            return int(float(value))  # 先转float处理小数，再转int
        elif target_type == float:
            return float(value)
        elif target_type == bool:
            # 处理非字符串类型的布尔值转换
            if isinstance(value, (int, float)):
                return value != 0
            return bool(value)
        else:  # 默认为字符串类型
            return str(value)
    except (ValueError, TypeError):
        logger.warning(f"⚠️ 无法转换值 {original_value} 到类型 {target_type}")
        return None


def parse_annotation_results(annotation_text: str) -> Tuple[List[str], Dict[str, float]]:
    """
    从人工注释结果中提取细胞类型列表和详细信息
    
    Args:
        annotation_text: 人工注释结果文本，如"Endothelial(38.6%),Fibroblast(26.34%)"
        
    Returns:
        tuple: (细胞类型列表, 细胞类型详细信息字典)
    """
    cell_types_list = []
    cell_type_details = {}
    
    if annotation_text:
        # 正则表达式匹配 'CellType(Percentage%)' 或 'CellType(Percentage)' 格式
        matches = re.findall(r'([a-zA-Z\s]+)\((\d+\.?\d*)%?\)', annotation_text)
        for cell_type, percentage in matches:
            cell_type = cell_type.strip()
            if cell_type and percentage:
                cell_types_list.append(cell_type)
                cell_type_details[cell_type] = float(percentage)
    
    return cell_types_list, cell_type_details


def generate_unique_id(prefix: str = "doc") -> str:
    """
    生成唯一ID，用于Milvus主键
    
    Args:
        prefix: ID前缀
        
    Returns:
        唯一ID字符串
    """
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ==========================================
# 常量定义 - 实验数据
# ==========================================
# 实验数据：原始字段到标准化字段的映射规则
EXPERIMENT_RAW_TO_STANDARDIZED_MAPPING = {
    # 单细胞平台
    "platform_type": ["项目建库类型", "project_library_type", "col_project_library_type", "实验平台", "平台", "col_实验平台", "col_平台", 
                      "col_project_library", "项目类型", "col_项目类型"],
    
    # 样本信息
    "species": ["物种", "species", "col_物种", "col_yang_ben_xin_xi"],
    "sample_type_exp": ["样本类型", "sample_type", "col_样本类型", "unnamed3"],
    "sample_detailed_type": ["样本详细类型", "sample_detailed_type", "col_样本详细类型", "组织", "组织类型", "col_组织", "col_组织类型", "tissue", "col_tissue", 
                             "unnamed3", "col_yang_ben_xin_xi_1"],
    "experiment_protocol": ["实验方案（解离/抽核）", "prep_method", "col_prep_method", "实验方案", "样本处理方式", "解离/抽核", "col_实验方案", "col_样本处理方式",
                            "col_yang_ben_xin_xi_2"],
    
    # 实验指标
        "arrival_temp_celsius": ["到样温度（℃）", "到样温度\n(℃)", "arrival_temp_celsius", "col_arrival_temp_celsius", "到样温度", "col_到样温度"],
        "total_cells_10k": ["细胞总量(万)", "细胞总量\n(万)", "total_cells_10k", "col_total_cells_10k", "细胞总量", "col_细胞总量", "col_xbzl_w", 
                            "col_shi_yan_zhi_biao", "col_yang_ben_xin_xi_3"],
    "clumping_rate_percent": ["结团率(%)", "clumping_rate_percent", "col_clumping_rate_percent", "结团率", "col_结团率", "col_jie_tuan_lv",
                              "col_shi_yan_zhi_biao_1"],
    "cell_viability_percent": ["细胞活率(%)", "cell_viability_percent", "col_cell_viability_percent", "细胞活率", "col_细胞活率", "col_xi_bao_huo_lv",
                               "col_shi_yan_zhi_biao_2"],
    "nucleated_rate_percent": ["有核率(%)", "nucleated_rate_percent", "col_nucleated_rate_percent", "有核率", "col_有核率", "col_you_he_lv",
                              "col_shi_yan_zhi_biao_3"],
    
    # 数据指标
    "captured_cells": ["捕获细胞数", "captured_cells", "col_captured_cells", "col_捕获细胞数", "col_bu_huo_xi_bao_shu",
                      "col_shu_ju_zhi_biao"],
    "reads_per_cell": ["reads/cell", "reads_per_cell", "col_reads_per_cell", "col_reads_cell",
                       "col_shu_ju_zhi_biao_1"],
    "median_genes": ["基因中位数", "median_genes", "col_median_genes", "col_基因中位数", "col_ji_yin_zhong_wei_shu",
                     "col_shu_ju_zhi_biao_2"],
    
    # 参考资料
    "annotation_results": ["人工注释结果", "annotation_results_full", "col_annotation_results_full", "col_人工注释结果", "col_zsjg_zztyxzs",
                           "col_shu_ju_zhi_biao_3"],
    "annotation_results_full": ["人工注释结果", "annotation_results_full", "col_annotation_results_full", "col_人工注释结果", "col_zsjg_zztyxzs",
                           "col_shu_ju_zhi_biao_3"],
    "tissue_digestion_protocol_name": ["组织消化方案", "tissue_digestion_protocol", "col_tissue_digestion_protocol", "col_组织消化方案"],
    "tissue_digestion_protocol_summary": ["组织消化方案概述", "tissue_digestion_summary", "col_tissue_digestion_summary", "col_组织消化方案概述"],
    "related_article_link": ["相关组织用户文章链接", "related_article_link", "col_related_article_link", "col_相关组织用户文章链接"],
    "feishu_doc_link": ["飞书文档链接", "feishu_doc_link", "col_feishu_doc_link", "col_飞书文档链接"],
    "video_stream_link": ["视频直播链接", "video_link", "col_video_link", "col_视频直播链接"],
    
    # 通用字段
    "project_id": ["project_id", "col_project_id", "项目编号", "col_项目编号"],
    "sample_date": ["sample_date", "col_sample_date", "送样日期", "col_送样日期"],
    "storage_method": ["storage_method", "col_storage_method", "样本保存方案", "col_样本保存方案"],
    "doc_id": ["doc_id", "col_doc_id", "文档ID", "col_文档ID"],
    "doc_type": ["doc_type", "col_doc_type", "文档类型", "col_文档类型"],
    "uploader": ["uploader", "col_uploader", "上传者", "col_上传者", "user_id", "col_user_id"],
    "sheet_name": ["sheet_name", "col_sheet_name", "工作表名称", "col_工作表名称"],
    "row_index": ["row_index", "col_row_index", "行索引", "col_行索引"],
    
    # 处理未知字段
    "unnamed1": ["unnamed1"],
    "unnamed2": ["unnamed2"],
    "unnamed3": ["unnamed3"],
    "unnamed4": ["unnamed4"],
    "unnamed5": ["unnamed5"],
}

# 实验数据：标准化字段类型映射
EXPERIMENT_STANDARDIZED_FIELD_TYPES = {
    # 单细胞平台
    "platform_type": str,
    
    # 样本信息
    "species": str,
    "sample_type_exp": str,
    "sample_detailed_type": str,
    "experiment_protocol": str,
    
    # 实验指标
    "arrival_temp_celsius": float,
    "total_cells_10k": float,
    "clumping_rate_percent": float,
    "cell_viability_percent": float,
    "nucleated_rate_percent": float,
    
    # 数据指标
    "captured_cells": int,
    "reads_per_cell": int,
    "median_genes": int,
    
    # 参考资料
    "annotation_results": str,
    "tissue_digestion_protocol_name": str,
    "tissue_digestion_protocol_summary": str,
    "related_article_link": str,
    "feishu_doc_link": str,
    "video_stream_link": str,
}

# ==========================================
# 常量定义 - 样本制备指南
# ==========================================
# 样本制备指南：原始字段到标准化字段的映射规则
PREPARATION_RAW_TO_STANDARDIZED_MAPPING = {
    # 产品信息
    "product_level1": [
        "产品一级目录", "product_level1", "col_product_level1", "col_产品一级目录",
        "产品一级", "一级目录", "col_产品一级", "col_一级目录"
    ],
    "product_level2": [
        "产品二级目录", "product_level2", "col_product_level2", "col_产品二级目录",
        "产品二级", "二级目录", "col_产品二级", "col_二级目录"
    ],
    "product_level3": [
        "产品三级目录", "product_level3", "col_product_level3", "col_产品三级目录",
        "产品三级", "三级目录", "col_产品三级", "col_三级目录"
    ],
    
    # 样本制备信息
    "sample_category_prep": [
        "样本大类", "sample_category", "col_样本大类", "样本大类", "category", "col_category",
        "样本类别", "col_样本类别"
    ],
    "sample_type_prep": [
        "样本类型", "sample_type", "col_样本类型", "col_样本类型",
        "样本类型细分", "col_样本类型细分", "样本类型_exp", "col_样本类型_exp"
    ],
    "tissue_type_prep": [
        "组织类型", "tissue_type", "col_tissue_type", "col_组织类型",
        "组织", "col_组织"
    ],
    "sample_prep_method": [
        "样本处理方式", "prep_method", "col_prep_method", "实验方案", "样本处理方式", 
        "解离/抽核", "col_实验方案", "col_样本处理方式"
    ],
    
    # 送样要求 - 带序号的字段映射
    "recommended_amount_1": ["建议送样量1", "col_建议送样量1", "建议送样量", "col_建议送样量"],
    "recommended_amount_2": ["建议送样量2", "col_建议送样量2", "建议送样量", "col_建议送样量"],
    "recommended_amount_3": ["建议送样量3", "col_建议送样量3", "建议送样量", "col_建议送样量"],
    "qualitative_description_1": ["定性描述1", "col_定性描述1", "定性描述", "col_定性描述"],
    "qualitative_description_2": ["定性描述2", "col_定性描述2", "定性描述", "col_定性描述"],
    "qualitative_description_3": ["定性描述3", "col_定性描述3", "定性描述", "col_定性描述"],
    
    # 方法和注意事项
    "sample_preparation_method_doc": [
        "样本准备方法", "preparation_method_doc", "col_preparation_method_doc", "col_样本准备方法",
        "样本制备方法", "col_样本制备方法"
    ],
    "sampling_notes": [
        "取样送样的注意事项", "handling_notes", "col_handling_notes", "col_取样送样的注意事项",
        "注意事项", "col_注意事项"
    ],
    "notes_full_text": [
        "备注", "notes_full", "col_notes_full", "col_备注",
        "说明", "col_说明"
    ],
    
    # 通用字段
    "doc_id": ["doc_id", "col_doc_id", "文档ID", "col_文档ID"],
    "doc_type": ["doc_type", "col_doc_type", "文档类型", "col_文档类型"],
    "uploader": ["uploader", "col_uploader", "上传者", "col_上传者", "user_id", "col_user_id"],
    "sheet_name": ["sheet_name", "col_sheet_name", "工作表名称", "col_工作表名称"],
    "row_index": ["row_index", "col_row_index", "行索引", "col_行索引"],
    
    # 处理未知字段
    "unnamed1": ["unnamed1"],
    "unnamed2": ["unnamed2"],
    "unnamed3": ["unnamed3"],
    "unnamed4": ["unnamed4"],
    "unnamed5": ["unnamed5"],
}

# 样本制备指南：标准化字段类型映射
PREPARATION_STANDARDIZED_FIELD_TYPES = {
    # 产品信息
    "product_level1": str,
    "product_level2": str,
    "product_level3": str,
    
    # 样本制备信息
    "sample_category_prep": str,
    "sample_type_prep": str,
    "tissue_type_prep": str,
    "sample_prep_method": str,
    
    # 送样要求 - 带序号的字段（均为字符串类型）
    "recommended_amount_1": str,
    "recommended_amount_2": str,
    "recommended_amount_3": str,
    "qualitative_description_1": str,
    "qualitative_description_2": str,
    "qualitative_description_3": str,
    
    # 方法和注意事项
    "sample_preparation_method_doc": str,
    "sampling_notes": str,
    "notes_full_text": str,
    
    # 通用字段
    "doc_id": str,
    "doc_type": str,
    "uploader": str,
    "sheet_name": str,
    "row_index": int,
    
    # 处理未知字段
    "unnamed1": str,
    "unnamed2": str,
    "unnamed3": str,
    "unnamed4": str,
    "unnamed5": str,
}

# ==========================================
# 文本模板定义
# ==========================================
# 实验数据文本模板
EXPERIMENT_TEXT_TEMPLATE = {
    "base": "该实验记录了单细胞平台类型为\"{platform_type}\"的样本。",
    "sample_info": "样本信息：物种为\"{species}\"，样本类型为\"{sample_type_exp}\"，详细类型是\"{sample_detailed_type}\"，实验方案采用\"{experiment_protocol}\"。",
    "experiment_metrics": "实验指标：到样温度为{arrival_temp_celsius}℃，细胞总量{total_cells_10k}万，结团率{clumping_rate_percent}%，细胞活率{cell_viability_percent}%，有核率{nucleated_rate_percent}%。",
    "data_metrics": "数据指标：捕获细胞数{captured_cells}，平均reads/cell为{reads_per_cell}，基因中位数为{median_genes}。",
    "annotation": "人工注释结果显示主要细胞类型及其占比为：{annotation_results}。",
    "reference": "组织消化方案：{tissue_digestion_protocol_name}，概述：{tissue_digestion_protocol_summary}。"
}

# 样本制备指南文本模板
PREPARATION_TEXT_TEMPLATE = {
    "base": "这是一条关于\"{product_level1}\"下\"{product_level3}\"产品的样本制备指南。",
    "sample_info": "样本属于\"{sample_category_prep}\"中的\"{sample_type_prep}\"，具体为\"{tissue_type_prep}\"组织，样本处理方式为\"{sample_prep_method}\"。",
    "sample_requirements": "送样要求：\n1. {recommended_amount_1}（{qualitative_description_1}）\n2. {recommended_amount_2}（{qualitative_description_2}）\n3. {recommended_amount_3}（{qualitative_description_3}）",
    "method_notes": "样本准备方法请参考文档：\"{sample_preparation_method_doc}\"，取样送样注意事项：{sampling_notes}，备注：{notes_full_text}。"
}


class DocumentTask(Task):
    """Celery 任务基类"""
    def on_failure(self, exc, task_id, args, kwargs, einfo):
        logger.error(f"❌ Task {task_id} failed: {exc}")
        super().on_failure(exc, task_id, args, kwargs, einfo)


@celery_app.task(
    bind=True, 
    base=DocumentTask, 
    name="tasks.process_document_async"  # 👈 修改任务名
)
def process_document_task_async(  # 👈 修改函数名
    self, 
    object_name: str, 
    filename: str, 
    department: str, 
    user_id: str = "system"
):
    """文档处理任务 - 使用异步 Embedding（优化版）"""
    
    logger.info(f"🚀 [AsyncTask] 开始处理文档: {filename}")
    logger.info(f"   部门: {department}, 用户: {user_id}")

    async def _async_workflow():
        try:
            # 1. 从 MinIO 下载
            logger.info("📥 [Step 1/8] 下载文件...")
            file_content = minio_client.download_file(object_name)
            logger.info(f"   文件大小: {len(file_content)} bytes")
            
            # 2. 获取解析器
            logger.info("🔍 [Step 2/8] 获取解析器...")
            processor = DocumentProcessorFactory.get_processor(
                filename, 
                department=department
            )
            if not processor:
                raise ValueError(f"不支持的文件格式: {filename}")
            
            # 3. 执行解析
            logger.info("📄 [Step 3/8] 解析文档...")
            process_result = await processor.process(
                file_content, 
                user_id=user_id, 
                filename=filename
            )
            
            # 4. 获取结果列表
            logger.info("🔄 [Step 4/8] 提取文档片段...")
            if hasattr(process_result, 'nodes'):
                raw_chunks = process_result.nodes
            elif hasattr(process_result, 'chunks'):
                raw_chunks = process_result.chunks
            elif hasattr(process_result, 'documents'):
                raw_chunks = process_result.documents
            else:
                logger.error(f"❌ ProcessResult 属性未知: {dir(process_result)}")
                raise AttributeError("ProcessResult 缺少 nodes/chunks/documents 属性")

            if not raw_chunks:
                logger.warning("⚠️ 文档解析结果为空")
                return {"status": "skipped", "message": "文档为空"}
                
            logger.info(f"✅ 解析完成，生成 {len(raw_chunks)} 个片段")
           
            # 5. 转换为 TextNode - 改进版本：数据预处理流水线
            logger.info("🔄 [Step 5/8] 转换为 TextNode...")
            nodes = []
            
            for i, chunk in enumerate(raw_chunks):
                # 每 500 个打印一次进度
                if i > 0 and i % 500 == 0:
                    logger.info(f"   进度: {i}/{len(raw_chunks)}")
                
                # 初始化数据容器
                current_raw_data = {}
                raw_text = ""
                
                if isinstance(chunk, TextNode):
                    # TextNode对象
                    current_raw_data = chunk.metadata.copy() if chunk.metadata else {}
                    raw_text = chunk.text
                else:
                    # 非TextNode对象，可能是ProcessedChunk或其他类型
                    # 1. 从metadata或extra_info提取
                    if hasattr(chunk, 'metadata'):
                        current_raw_data.update(chunk.metadata.copy() if chunk.metadata else {})
                    if hasattr(chunk, 'extra_info'):
                        current_raw_data.update(chunk.extra_info.copy() if chunk.extra_info else {})
                    
                    # 2. 直接从chunk对象属性提取所有字段（除了内置属性和方法）
                    # 优化：先获取所有属性名，包括动态添加的属性
                    attr_names = set(dir(chunk))
                    
                    # 添加可能的动态属性名
                    potential_attrs = ['department', 'filename', 'chunk_id', 'owner', 'table_type', 
                                      'project_library_type', 'species', 'sample_detailed_type', 
                                      'total_cells_10k', 'captured_cells', 'unnamed3', 'sample_type',
                                      'prep_method', 'arrival_temp_celsius', 'clumping_rate_percent',
                                      'cell_viability_percent', 'nucleated_rate_percent', 'reads_per_cell',
                                      'median_genes', 'annotation_results_full', 'product_level1',
                                      'product_level3', 'sample_category', 'recommended_amount_mg',
                                      'risk_level', 'preparation_method_doc', 'handling_notes']
                    
                    attr_names.update(potential_attrs)
                    
                    for attr_name in attr_names:
                        if not attr_name.startswith('_') and attr_name not in ['metadata', 'extra_info', 'text', 'content', 'id', 'node_id', 'ref_doc_id']:
                            try:
                                attr_value = getattr(chunk, attr_name)
                                # 只添加非函数和非内置类型的属性
                                if not callable(attr_value) and not isinstance(attr_value, (type, object)):
                                    current_raw_data[attr_name] = attr_value
                            except Exception as e:
                                # 只记录可能重要的属性获取失败
                                if attr_name in potential_attrs:
                                    logger.debug(f"⚠️ 无法获取重要属性 {attr_name}: {e}")
                    
                    # 获取文本内容
                    if hasattr(chunk, 'text'):
                        raw_text = str(chunk.text)
                    elif hasattr(chunk, 'content'):
                        raw_text = str(chunk.content)
                    else:
                        raw_text = str(chunk)
                
                # === 数据预处理流水线 ===
                # 1. 动态判断表格类型
                table_type = "general"
                
                # 方法0: 优先根据sheet_name判断表格类型（新增）
                explicit_table_type = None
                sheet_name = current_raw_data.get("sheet_name", "").lower()
                
                # 直接根据sheet_name判断表格类型
                if sheet_name:
                    if any(keyword in sheet_name for keyword in ["经验", "实验", "experimental", "exp"]):
                        explicit_table_type = "experiment_data"
                    elif any(keyword in sheet_name for keyword in ["样本类型", "保存方式", "制备", "preparation", "guide"]):
                        explicit_table_type = "preparation_guidelines"
                
                # 方法1: 检查是否有明确的表格类型字段
                if not explicit_table_type and any(table_field in current_raw_data for table_field in ["source_table", "table_type", "表类型"]):
                    table_type_value = current_raw_data.get("source_table", "") or current_raw_data.get("table_type", "") or current_raw_data.get("表类型", "")
                    if "实验" in table_type_value or "experimental" in table_type_value.lower():
                        explicit_table_type = "experiment_data"
                    elif "制备" in table_type_value or "preparation" in table_type_value.lower() or "guide" in table_type_value.lower():
                        explicit_table_type = "preparation_guidelines"
                
                # 方法2: 如果仍无法确定，根据关键字段数量判断
                if not explicit_table_type:
                    # 检查是否为实验数据（同时检查中英文字段）
                    experiment_key_fields = [
                        # 中文关键字段
                        "细胞总量(万)", "结团率(%)", "细胞活率(%)", "捕获细胞数", "reads/cell", "基因中位数",
                        # 英文关键字段
                        "project_library_type", "species", "sample_detailed_type", "total_cells_10k", "captured_cells", "reads_per_cell", "median_genes"
                    ]
                    experiment_field_count = sum(1 for field in experiment_key_fields if field in current_raw_data)
                    
                    # 检查是否为样本制备指南（同时检查中英文字段）
                    preparation_key_fields = [
                        # 中文关键字段
                        "产品一级目录", "产品三级目录", "样本准备方法", "推荐送样量", "取样送样的注意事项",
                        # 英文关键字段
                        "product_level1", "product_level3", "preparation_method_doc", "recommended_amount_mg", "handling_notes",
                        # 特殊表格字段
                        "qualitative_description_text", "risk_level"
                    ]
                    preparation_field_count = sum(1 for field in preparation_key_fields if field in current_raw_data)
                    
                    # 根据匹配字段数量决定表格类型
                    if experiment_field_count > preparation_field_count and experiment_field_count >= 2:
                        explicit_table_type = "experiment_data"
                    elif preparation_field_count > experiment_field_count and preparation_field_count >= 2:
                        explicit_table_type = "preparation_guidelines"
                
                # 应用判断结果
                if explicit_table_type:
                    table_type = explicit_table_type
                
                logger.debug(f"📋 识别表格类型: {table_type} 用于chunk {i}")
                
                # 2. 从原始数据构建标准化元数据
                standardized_metadata = {
                    "department": department,
                    "filename": filename,
                    "doc_id": filename,  # 使用文件名作为默认 doc_id
                    "doc_type": "experiment" if table_type == "experiment_data" else "preparation",  # 根据表格类型设置文档类型
                    "chunk_id": int(i),
                    "owner": user_id,
                    "uploader": user_id,  # uploader 来自 user_id 参数
                    "raw_text": raw_text,  # 保留原始文本，便于调试
                    "source_table": table_type,  # 添加表格类型到元数据，统一使用 source_table
                    # 根据表格类型设置必填字段的默认值，只包含当前表格类型相关的字段
                    **({  # 实验数据表格必填字段
                        "platform_type": "unknown",
                        "species": "",
                        "sample_type_exp": "",
                        "sample_detailed_type": "",
                        "experiment_protocol": ""
                    } if table_type == "experiment_data" else {  # 样本制备指南表格必填字段
                        "product_level1": "",
                        "product_level2": "",
                        "product_level3": "",
                        "sample_category_prep": "",
                        "sample_type_prep": "",
                        "tissue_type_prep": "",
                        "sample_prep_method": ""
                    })
                }
                
                # 3. 根据表格类型选择相应的映射规则和字段类型
                if table_type == "experiment_data":
                    mapping = EXPERIMENT_RAW_TO_STANDARDIZED_MAPPING
                    field_types = EXPERIMENT_STANDARDIZED_FIELD_TYPES
                elif table_type == "preparation_guidelines":
                    mapping = PREPARATION_RAW_TO_STANDARDIZED_MAPPING
                    field_types = PREPARATION_STANDARDIZED_FIELD_TYPES
                else:
                    # 通用映射，包含所有字段
                    mapping = {**EXPERIMENT_RAW_TO_STANDARDIZED_MAPPING, **PREPARATION_RAW_TO_STANDARDIZED_MAPPING}
                    field_types = {**EXPERIMENT_STANDARDIZED_FIELD_TYPES, **PREPARATION_STANDARDIZED_FIELD_TYPES}
                
                # 4. 应用映射规则，将原始字段转换为标准化字段
                for std_field, raw_fields in mapping.items():
                    field_value = None
                    
                    # 首先检查原始数据中是否已经包含标准化字段名，并且值不为空
                    if std_field in current_raw_data:
                        raw_value = current_raw_data[std_field]
                        target_type = field_types.get(std_field, str)
                        field_value = clean_and_convert_value(raw_value, target_type)
                    
                    # 如果标准化字段值为空，按优先级检查原始字段
                    if field_value is None:
                        for raw_field in raw_fields:
                            if raw_field in current_raw_data:
                                raw_value = current_raw_data[raw_field]
                                target_type = field_types.get(std_field, str)
                                field_value = clean_and_convert_value(raw_value, target_type)
                                
                                if field_value is not None:
                                    break
                    
                    # 总是添加字段，即使值为None，以便后续处理
                    standardized_metadata[std_field] = field_value
                
                # 5. 直接映射excel_processor.py生成的字段名到Milvus字段名
                # 解决excel_processor.py和document_tasks_async.py之间的字段名不匹配问题
                # 处理样本类型映射
                if "sample_type" in current_raw_data and current_raw_data["sample_type"]:
                    standardized_metadata["sample_type_exp"] = current_raw_data["sample_type"]
                    # 不添加sample_type字段，避免重复
                
                # 处理实验平台映射
                if "platform" in current_raw_data and current_raw_data["platform"]:
                    standardized_metadata["platform_type"] = current_raw_data["platform"]
                
                # 处理实验方案映射
                if "实验方案" in current_raw_data and current_raw_data["实验方案"]:
                    standardized_metadata["experiment_protocol"] = current_raw_data["实验方案"]
                
                # 处理人工细胞注释映射
                if "annotation_results" in current_raw_data and current_raw_data["annotation_results"]:
                    standardized_metadata["annotation_results"] = current_raw_data["annotation_results"]
                    # 只保留一个注释字段，避免重复
                elif "annotation_results_full" in current_raw_data and current_raw_data["annotation_results_full"]:
                    standardized_metadata["annotation_results"] = current_raw_data["annotation_results_full"]
                elif "人工细胞注释" in current_raw_data and current_raw_data["人工细胞注释"]:
                    standardized_metadata["annotation_results"] = current_raw_data["人工细胞注释"]
                
                # 处理遗留字段 - 删除
                for legacy_field in ["is_lysis", "is_dead_removal"]:
                    if legacy_field in current_raw_data:
                        del current_raw_data[legacy_field]
                
                # 从full_row_json中提取字段
                if "full_row_json" in current_raw_data:
                    try:
                        full_row = json.loads(current_raw_data["full_row_json"])
                        
                        # 根据表格类型提取相应的字段
                        if table_type == "experiment_data":
                            # 实验数据表格字段提取
                            if "人工细胞注释" in full_row:
                                standardized_metadata["annotation_results"] = full_row["人工细胞注释"]
                            if "实验平台" in full_row:
                                standardized_metadata["platform_type"] = full_row["实验平台"]
                            if "实验方案" in full_row:
                                standardized_metadata["experiment_protocol"] = full_row["实验方案"]
                            
                            # 实验指标字段
                            if "到样温度\n(℃)" in full_row:
                                standardized_metadata["arrival_temp_celsius"] = full_row["到样温度\n(℃)"]
                            if "细胞总量\n(万)" in full_row:
                                standardized_metadata["total_cells_10k"] = full_row["细胞总量\n(万)"]
                            elif "细胞总量(万)" in full_row:
                                standardized_metadata["total_cells_10k"] = full_row["细胞总量(万)"]
                            if "结团率(%)" in full_row:
                                standardized_metadata["clumping_rate_percent"] = full_row["结团率(%)"]
                            if "细胞活率(%)" in full_row:
                                standardized_metadata["cell_viability_percent"] = full_row["细胞活率(%)"]
                            if "有核率(%)" in full_row:
                                standardized_metadata["nucleated_rate_percent"] = full_row["有核率(%)"]
                            if "捕获细胞数" in full_row:
                                standardized_metadata["captured_cells"] = full_row["捕获细胞数"]
                            if "reads/cell" in full_row:
                                standardized_metadata["reads_per_cell"] = full_row["reads/cell"]
                            if "基因中位数" in full_row:
                                standardized_metadata["median_genes"] = full_row["基因中位数"]
                        elif table_type == "preparation_guidelines":
                            # 样本制备指南表格字段提取
                            # 产品信息字段
                            if "产品一级目录" in full_row:
                                standardized_metadata["product_level1"] = full_row["产品一级目录"]
                            if "产品二级目录" in full_row:
                                standardized_metadata["product_level2"] = full_row["产品二级目录"]
                            if "产品三级目录" in full_row:
                                standardized_metadata["product_level3"] = full_row["产品三级目录"]
                            
                            # 样本信息字段
                            if "样本大类" in full_row:
                                standardized_metadata["sample_category_prep"] = full_row["样本大类"]
                            if "样本类型" in full_row:
                                standardized_metadata["sample_type_prep"] = full_row["样本类型"]
                            if "组织类型" in full_row:
                                standardized_metadata["tissue_type_prep"] = full_row["组织类型"]
                            if "样本处理方式" in full_row:
                                standardized_metadata["sample_prep_method"] = full_row["样本处理方式"]
                            
                            # 送样要求字段
                            if "建议送样量1" in full_row:
                                standardized_metadata["recommended_amount_1"] = full_row["建议送样量1"]
                            if "建议送样量2" in full_row:
                                standardized_metadata["recommended_amount_2"] = full_row["建议送样量2"]
                            if "建议送样量3" in full_row:
                                standardized_metadata["recommended_amount_3"] = full_row["建议送样量3"]
                            if "定性描述1" in full_row:
                                standardized_metadata["qualitative_description_1"] = full_row["定性描述1"]
                            if "定性描述2" in full_row:
                                standardized_metadata["qualitative_description_2"] = full_row["定性描述2"]
                            if "定性描述3" in full_row:
                                standardized_metadata["qualitative_description_3"] = full_row["定性描述3"]
                            
                            # 方法和注意事项
                            if "样本准备方法" in full_row:
                                standardized_metadata["sample_preparation_method_doc"] = full_row["样本准备方法"]
                            if "取样送样的注意事项" in full_row:
                                standardized_metadata["sampling_notes"] = full_row["取样送样的注意事项"]
                            if "备注" in full_row:
                                standardized_metadata["notes_full_text"] = full_row["备注"]
                        
                        # 提取其他通用字段
                        if "组织重量\n（数值）" in full_row and full_row["组织重量\n（数值）"] and full_row["组织重量\n（数值）"] != "/":
                            standardized_metadata["tissue_weight"] = full_row["组织重量\n（数值）"]
                    except json.JSONDecodeError:
                        pass
                
                # 5. 添加所有未被映射但存在的字段，确保不丢失信息
                for raw_field, raw_value in current_raw_data.items():
                    if raw_field not in standardized_metadata:
                        # 只添加有值的字段
                        if raw_value is not None and raw_value != "":
                            # 只处理可序列化的类型
                            if isinstance(raw_value, (str, int, float, bool, list, dict)):
                                # 确保列表和字典中的元素也是可序列化的
                                if isinstance(raw_value, (list, dict)):
                                    try:
                                        # 测试序列化
                                        json.dumps(raw_value)
                                        standardized_metadata[raw_field] = raw_value
                                    except (TypeError, ValueError):
                                        # 无法序列化，转换为字符串
                                        standardized_metadata[raw_field] = str(raw_value)
                                else:
                                    standardized_metadata[raw_field] = raw_value
                            elif hasattr(raw_value, '__str__'):
                                # 其他类型尝试转换为字符串
                                standardized_metadata[raw_field] = str(raw_value)
                
                # 5. 处理人工注释结果（仅适用于实验数据）
                if table_type == "experiment_data" and "annotation_results" in standardized_metadata:
                    annotation_text = standardized_metadata["annotation_results"]
                    cell_types_list, cell_type_details = parse_annotation_results(annotation_text)
                    # 注：新的Schema中不再包含annotation_cell_types_list和annotation_cell_type_details字段
                    # 这些字段已被annotation_results替代
                    # 此处保留代码但注释掉，以便后续可能需要恢复
                    # if cell_types_list:
                    #     standardized_metadata["annotation_cell_types_list"] = cell_types_list
                    # if cell_type_details:
                    #     standardized_metadata["annotation_cell_type_details"] = cell_type_details
                
                # 6. 生成唯一ID
                milvus_id = generate_unique_id(prefix="milvus")
                standardized_metadata["id"] = milvus_id
                
                # 7. 确保所有Milvus Schema字段都存在于标准化元数据中
                # 只包含核心RAG字段和公共元数据字段，其他字段作为动态字段处理
                schema_fields = {
                    # 核心RAG字段
                    "doc_id": "",
                    "text": "",  # text会在后面单独设置
                    "embedding": None,  # embedding会在后面单独设置
                    
                    # 公共元数据字段
                    "source_table": "",
                    "chunk_id": None,
                    "filename": "",
                    "department": "",
                    "doc_type": "",
                    "owner": "",
                    "uploader": "",
                    "sheet_name": "",
                    "row_index": None,
                }
                
                # 添加所有缺失的Schema字段，使用None作为默认值
                for field_name, default_value in schema_fields.items():
                    if field_name not in standardized_metadata:
                        standardized_metadata[field_name] = default_value
                
                # 8. 处理数值类型字段：确保数值类型字段是正确的类型，避免Milvus插入错误
                # 只处理公共字段的数值类型
                numeric_fields = {
                    "row_index": 0,
                    "chunk_id": 0
                }
                
                for field_name in standardized_metadata:
                    # 确保数值类型字段是正确的类型
                    if field_name in numeric_fields:
                        value = standardized_metadata[field_name]
                        # 显式转换为整数类型
                        try:
                            standardized_metadata[field_name] = int(value) if value is not None else numeric_fields[field_name]
                        except (ValueError, TypeError):
                            # 如果转换失败，使用默认值
                            standardized_metadata[field_name] = numeric_fields[field_name]
                
                # 🔍 确保annotation_results字段始终包含正确的值
                # 这是修复annotation_results字段为空的关键
                if standardized_metadata.get("annotation_results") is None:
                    if standardized_metadata.get("annotation_results_full"):
                        standardized_metadata["annotation_results"] = standardized_metadata["annotation_results_full"]
                    # 最后尝试从full_row_json中提取
                    elif "full_row_json" in current_raw_data:
                        try:
                            full_row = json.loads(current_raw_data["full_row_json"])
                            if "人工细胞注释" in full_row and full_row["人工细胞注释"] and full_row["人工细胞注释"] != "/":
                                standardized_metadata["annotation_results"] = full_row["人工细胞注释"]
                                standardized_metadata["annotation_results_full"] = full_row["人工细胞注释"]
                        except json.JSONDecodeError:
                            pass
                
                # 9. 处理字符串类型字段的None值，转换为空字符串
                # 只处理核心RAG字段和公共元数据字段
                string_fields = [
                    # 核心RAG字段
                    "doc_id",
                    "text",
                    
                    # 公共元数据字段
                    "source_table",
                    "filename",
                    "department",
                    "doc_type",
                    "owner",
                    "uploader",
                    "sheet_name",
                ]
                
                for field_name in standardized_metadata:
                    # 只处理核心RAG字段和公共元数据字段
                    if field_name in string_fields and standardized_metadata[field_name] is None:
                        standardized_metadata[field_name] = ""
                
                # 10. 所有字段均为字符串类型，不需要特殊处理数组字段
                
                # 7. 动态构建TextNode.text内容
                # 严格根据表格类型构建文本，只使用当前表格类型相关的字段
                text_parts = []
                
                if table_type == "experiment_data":
                    # 实验数据文本构建
                    if "platform_type" in standardized_metadata:
                        text_parts.append(EXPERIMENT_TEXT_TEMPLATE["base"].format_map({"platform_type": standardized_metadata["platform_type"]}))
                    
                    # 样本信息
                    sample_info_fields = {}
                    for field in ["species", "sample_type_exp", "sample_detailed_type", "experiment_protocol"]:
                        if field in standardized_metadata:
                            # 当字段值为空字符串或None时，显示为"未知"
                            value = standardized_metadata[field]
                            sample_info_fields[field] = value if value is not None and value.strip() else "未知"
                    if sample_info_fields:
                        # 使用安全格式化，只替换存在的字段
                        safe_template = EXPERIMENT_TEXT_TEMPLATE["sample_info"]
                        for field in ["species", "sample_type_exp", "sample_detailed_type", "experiment_protocol"]:
                            if field not in sample_info_fields:
                                safe_template = safe_template.replace(f"{{{field}}}", "未知")
                        text_parts.append(safe_template.format_map(sample_info_fields))
                    
                    # 实验指标
                    exp_metrics_fields = {}
                    for field in ["arrival_temp_celsius", "total_cells_10k", "clumping_rate_percent", "cell_viability_percent", "nucleated_rate_percent"]:
                        if field in standardized_metadata:
                            exp_metrics_fields[field] = standardized_metadata[field]
                    if exp_metrics_fields:
                        safe_template = EXPERIMENT_TEXT_TEMPLATE["experiment_metrics"]
                        for field in ["arrival_temp_celsius", "total_cells_10k", "clumping_rate_percent", "cell_viability_percent", "nucleated_rate_percent"]:
                            if field not in exp_metrics_fields:
                                safe_template = safe_template.replace(f"{{{field}}}", "未知")
                        text_parts.append(safe_template.format_map(exp_metrics_fields))
                    
                    # 数据指标
                    data_metrics_fields = {}
                    for field in ["captured_cells", "reads_per_cell", "median_genes"]:
                        if field in standardized_metadata:
                            data_metrics_fields[field] = standardized_metadata[field]
                    if data_metrics_fields:
                        safe_template = EXPERIMENT_TEXT_TEMPLATE["data_metrics"]
                        for field in ["captured_cells", "reads_per_cell", "median_genes"]:
                            if field not in data_metrics_fields:
                                safe_template = safe_template.replace(f"{{{field}}}", "未知")
                        text_parts.append(safe_template.format_map(data_metrics_fields))
                    
                    # 人工注释结果
                    if "annotation_results" in standardized_metadata:
                        text_parts.append(EXPERIMENT_TEXT_TEMPLATE["annotation"].format_map({"annotation_results": standardized_metadata["annotation_results"]}))
                
                elif table_type == "preparation_guidelines":
                    # 样本制备指南文本构建
                    if all(field in standardized_metadata for field in ["product_level1", "product_level3"]):
                        text_parts.append(PREPARATION_TEXT_TEMPLATE["base"].format_map({"product_level1": standardized_metadata["product_level1"], "product_level3": standardized_metadata["product_level3"]}))
                    
                    # 样本信息
                    sample_info_fields = {}
                    for field in ["sample_category_prep", "sample_type_prep", "tissue_type_prep", "sample_prep_method"]:
                        if field in standardized_metadata:
                            sample_info_fields[field] = standardized_metadata[field]
                    if sample_info_fields:
                        safe_template = PREPARATION_TEXT_TEMPLATE["sample_info"]
                        for field in ["sample_category_prep", "sample_type_prep", "tissue_type_prep", "sample_prep_method"]:
                            if field not in sample_info_fields:
                                safe_template = safe_template.replace(f"{{{field}}}", "未知")
                        text_parts.append(safe_template.format_map(sample_info_fields))
                    
                    # 送样要求 - 使用编号字段
                    sample_req_fields = {}
                    for field in ["recommended_amount_1", "recommended_amount_2", "recommended_amount_3", 
                                 "qualitative_description_1", "qualitative_description_2", "qualitative_description_3"]:
                        if field in standardized_metadata:
                            sample_req_fields[field] = standardized_metadata[field]
                    if sample_req_fields:
                        safe_template = PREPARATION_TEXT_TEMPLATE["sample_requirements"]
                        for field in ["recommended_amount_1", "recommended_amount_2", "recommended_amount_3", 
                                     "qualitative_description_1", "qualitative_description_2", "qualitative_description_3"]:
                            if field not in sample_req_fields:
                                safe_template = safe_template.replace(f"{{{field}}}", "")
                        text_parts.append(safe_template.format_map(sample_req_fields))
                    
                    # 方法和注意事项
                    method_notes_fields = {}
                    for field in ["sample_preparation_method_doc", "sampling_notes", "notes_full_text"]:
                        if field in standardized_metadata:
                            method_notes_fields[field] = standardized_metadata[field]
                    if method_notes_fields:
                        safe_template = PREPARATION_TEXT_TEMPLATE["method_notes"]
                        for field in ["sample_preparation_method_doc", "sampling_notes", "notes_full_text"]:
                            if field not in method_notes_fields:
                                safe_template = safe_template.replace(f"{{{field}}}", "未知")
                        text_parts.append(safe_template.format_map(method_notes_fields))
                
                # 通用文本生成：如果没有匹配的模板，生成基本描述，只使用当前表格类型相关的字段
                if not text_parts:
                    # 生成包含关键元数据的基本描述，只使用当前表格类型相关的字段
                    key_metadata = []
                    for key, value in standardized_metadata.items():
                        if key not in ["department", "filename", "chunk_id", "owner", "raw_text", "id", "table_type"] and value is not None and value != "":
                            key_metadata.append(f"{key}: {value}")
                    if key_metadata:
                        text_parts.append("\n".join(key_metadata))
                    else:
                        # 使用原始文本作为最后兜底
                        text_parts.append(raw_text or "无内容")
                
                # 最终文本生成
                node_text = "\n".join(text_parts)
                
                # 确保文本不为空，根据表格类型显示不同的默认文本
                if not node_text or node_text.strip() == "":
                    if table_type == "experiment_data":
                        node_text = "该文档包含实验数据：" + ", ".join([f"{k}={v}" for k, v in standardized_metadata.items() if k not in ["department", "filename", "chunk_id", "owner", "raw_text", "id", "table_type"] and v is not None and v != ""])
                    else:
                        node_text = "该文档包含样本制备指南：" + ", ".join([f"{k}={v}" for k, v in standardized_metadata.items() if k not in ["department", "filename", "chunk_id", "owner", "raw_text", "id", "table_type"] and v is not None and v != ""])
                
                # 6. 构建最终的元数据（移除临时字段、冗余字段和无用字段）
                final_metadata = standardized_metadata.copy()
                
                # 移除临时字段
                if "raw_text" in final_metadata:
                    del final_metadata["raw_text"]
                
                # 移除所有以"col_"开头的冗余字段，只保留标准化字段
                keys_to_remove = [key for key in final_metadata if key.startswith("col_")]
                
                # 移除重复字段和遗留字段
                keys_to_remove.extend([
                    "sample_type",  # 与sample_type_exp重复
                    "annotation_results_full",  # 与annotation_results重复
                    "is_lysis",  # 遗留字段
                    "is_dead_removal",  # 遗留字段
                    "rin_score",  # 遗留字段
                ])
                
                # 移除无用的null值字段
                keys_to_remove.extend([
                    "project_id",
                    "sample_date",
                    "storage_method",
                    "unnamed1",
                    "unnamed2",
                    "unnamed3",
                    "unnamed4",
                    "unnamed5",
                ])
                
                # 移除文档ID相关字段
                keys_to_remove.extend([
                    "document_id",
                    "ref_doc_id",
                ])
                
                # 根据表格类型移除不相关的字段
                if table_type == "experiment_data":
                    # 移除样本制备指南相关字段
                    keys_to_remove.extend([
                        "product_level1", "product_level2", "product_level3",
                        "sample_category_prep", "sample_type_prep", "tissue_type_prep",
                        "sample_prep_method", "recommended_amount_1", "recommended_amount_2",
                        "recommended_amount_3", "qualitative_description_1", "qualitative_description_2",
                        "qualitative_description_3", "sample_preparation_method_doc",
                        "sampling_notes", "notes_full_text"
                    ])
                else:
                    # 移除实验数据相关字段
                    keys_to_remove.extend([
                        "platform_type", "species", "sample_type_exp", "sample_detailed_type",
                        "experiment_protocol", "arrival_temp_celsius", "total_cells_10k",
                        "clumping_rate_percent", "cell_viability_percent", "nucleated_rate_percent",
                        "captured_cells", "reads_per_cell", "median_genes", "annotation_results",
                        "tissue_digestion_protocol_name", "tissue_digestion_protocol_summary",
                        "related_article_link", "feishu_doc_link", "video_stream_link"
                    ])
                
                # 执行移除操作
                for key in keys_to_remove:
                    if key in final_metadata:
                        del final_metadata[key]
                
                # 额外清理：移除所有值为None且不属于核心字段的字段
                core_fields = [
                    "doc_id", "text", "embedding", "source_table", "chunk_id", 
                    "filename", "department", "doc_type", "owner", "uploader", 
                    "sheet_name", "row_index"
                ]
                
                none_keys_to_remove = [
                    key for key in final_metadata 
                    if final_metadata[key] is None 
                    and key not in core_fields
                ]
                
                for key in none_keys_to_remove:
                    if key in final_metadata:
                        del final_metadata[key]
                
                # 7. 确保元数据可序列化
                # 使用内置的序列化方法，避免json模块作用域问题
                for key, value in list(final_metadata.items()):
                    try:
                        # 使用repr()和eval()作为替代序列化检查
                        repr(value)
                    except Exception as e:
                        logger.warning(f"⚠️ 元数据 {key} 无法序列化，转换为字符串: {e}")
                        final_metadata[key] = str(value)
                
                # 8. 创建TextNode对象
                node = TextNode(
                    text=node_text,
                    metadata=final_metadata,
                    excluded_embed_metadata_keys=[
                        "chunk_id", "filename", "department", "owner", "id"
                    ]
                )
                
                nodes.append(node)
            
            logger.info(f"✅ 转换完成 {len(nodes)} 个节点")
            
            # 6. 使用异步接口批量生成 Embeddings
            logger.info("🚀 [Step 6/8] 异步生成 Embeddings...")
            logger.info(f"   模型: text-embedding-async-v2")
            logger.info(f"   数量: {len(nodes)} 个文本")
            
            async_embed_client = get_async_embed_model(
                model_name="text-embedding-v4"
            )
            
            texts = [node.text for node in nodes]
            
            # 一次性提交所有文本（关键优化！）
            embeddings = await async_embed_client.embed_texts(texts)
            
            logger.info(f"✅ Embeddings 生成完成，共 {len(embeddings)} 个向量")
            
            # 7. 将 embeddings 附加到 nodes
            logger.info("🔗 [Step 7/8] 附加向量到节点...")
            for node, embedding in zip(nodes, embeddings):
                node.embedding = embedding
            
            # 8. 直接写入 Milvus
            logger.info("💾 [Step 8/8] 写入 Milvus...")
            milvus_manager.insert_nodes_with_embeddings(
                department=department,
                nodes=nodes,
                show_progress=True
            )
            
            logger.info(f"🎉 处理完成! 成功写入 {len(nodes)} 个节点到 {department}")
            
            # 修复：确保返回的结果可以被JSON序列化
            # 避免pandas int64/float64类型导致的序列化错误
            result = {
                "status": "success", 
                "filename": filename, 
                "nodes_count": int(len(nodes)),
                "department": department
            }
            
            # 🔧 最终防护：确保结果是可序列化的
            try:
                # 使用已导入的json模块
                json.dumps(result)
                return result
            except Exception as e:
                logger.warning(f"⚠️ 结果无法直接序列化，转换为字符串: {e}")
                return {
                    "status": "success",
                    "filename": str(filename),
                    "nodes_count": int(len(nodes)),
                    "department": str(department),
                    "warning": f"部分结果已转换为字符串: {e}"
                }

        except Exception as e:
            logger.exception(f"❌ [AsyncTask] 处理失败: {filename}")
            raise e

    # 运行异步工作流
    try:
        return asyncio.run(_async_workflow())
    except Exception as e:
        logger.exception(f"❌ [AsyncTask] 任务执行异常")
        raise e
