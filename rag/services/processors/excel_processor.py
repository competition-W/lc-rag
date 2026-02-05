#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
import json
import uuid
import re
from typing import List, Dict, Any, Optional
import pandas as pd
from pypinyin import lazy_pinyin, Style

from llama_index.core import Document
from llama_index.core.schema import TextNode
# 假设您的 schema_manager 在这里，如果路径不对请调整
from services.schema_manager import schema_manager

from .base import BaseDocumentProcessor, ProcessResult
from config import settings
import logging

logger = logging.getLogger(__name__)

# =========================================================
# 🔥 1. 核心定义：中英文列名映射表
# 从公共字段映射模块导入，确保系统中使用统一的字段映射
# =========================================================
from services.common.field_mapping import COLUMN_MAPPING

# 动态映射表，用于存储自动生成的映射
DYNAMIC_COLUMN_MAPPING = {}

# 映射表文件路径
MAPPING_FILE_PATH = "data/column_mapping.json"

class ExcelProcessor(BaseDocumentProcessor):
    """Excel处理器（LlamaIndex集成版 - 已修复Key编码问题）"""
    
    def __init__(self, department: str):
        super().__init__(department)
        self.skip_rows = settings.EXCEL_SKIP_HEADER_ROWS
        # 加载动态映射表
        self._load_dynamic_mapping()
    
    def _load_dynamic_mapping(self):
        """加载动态映射表"""
        global DYNAMIC_COLUMN_MAPPING
        import os
        
        # 确保data目录存在
        data_dir = os.path.dirname(MAPPING_FILE_PATH)
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # 加载映射表
        if os.path.exists(MAPPING_FILE_PATH):
            try:
                with open(MAPPING_FILE_PATH, 'r', encoding='utf-8') as f:
                    DYNAMIC_COLUMN_MAPPING = json.load(f)
                logger.info(f"✅ 加载动态映射表成功，共 {len(DYNAMIC_COLUMN_MAPPING)} 个映射")
            except Exception as e:
                logger.warning(f"⚠️ 加载动态映射表失败: {e}")
                DYNAMIC_COLUMN_MAPPING = {}
    
    def _save_dynamic_mapping(self):
        """保存动态映射表"""
        import os
        
        try:
            with open(MAPPING_FILE_PATH, 'w', encoding='utf-8') as f:
                json.dump(DYNAMIC_COLUMN_MAPPING, f, ensure_ascii=False, indent=2)
            logger.info(f"✅ 保存动态映射表成功，共 {len(DYNAMIC_COLUMN_MAPPING)} 个映射")
        except Exception as e:
            logger.error(f"❌ 保存动态映射表失败: {e}")
    
    def _update_dynamic_mapping(self, original_col: str, mapped_col: str):
        """更新动态映射表"""
        global DYNAMIC_COLUMN_MAPPING
        
        # 只有当原始列名不在预定义映射表中时，才保存到动态映射表
        if original_col not in COLUMN_MAPPING:
            if original_col not in DYNAMIC_COLUMN_MAPPING or DYNAMIC_COLUMN_MAPPING[original_col] != mapped_col:
                DYNAMIC_COLUMN_MAPPING[original_col] = mapped_col
                logger.info(f"📝 更新动态映射: {original_col} -> {mapped_col}")
                # 保存映射表
                self._save_dynamic_mapping()
    
    def validate_file(self, file_data: bytes, filename: str) -> bool:
        try:
            io.BytesIO(file_data)
            return filename.endswith(('.xlsx', '.xls', '.xlsm'))
        except Exception as e:
            self.logger.error(f"文件验证失败: {e}")
            return False
    
    async def process(
        self,
        file_data: bytes,
        filename: str,
        user_id: str,
        sheet_name: Optional[str] = None,
        **kwargs
    ) -> ProcessResult:
        try:
            file_obj = io.BytesIO(file_data)
            
            # 读取Excel - 使用第2行作为列名（处理多级表头）
            if sheet_name:
                df_dict = {sheet_name: pd.read_excel(file_obj, sheet_name=sheet_name, header=1)}
            else:
                df_dict = pd.read_excel(file_obj, sheet_name=None, header=1)
            
            # ==========================================
            # 🟢 处理重复表头：将重复的二级表头编号（如"建议送样量1"、"定性描述2"）
            # ==========================================
            for sheet, df in df_dict.items():
                # 统计每个表头出现的次数
                header_count = {}
                new_columns = []
                
                for col in df.columns:
                    col_str = str(col).strip()
                    
                    # 处理重复表头：只处理"建议送样量"和"定性描述"这两个表头
                    if "建议送样量" in col_str:
                        base_header = "建议送样量"
                    elif "定性描述" in col_str:
                        base_header = "定性描述"
                    else:
                        # 其他表头不处理
                        new_columns.append(col_str)
                        continue
                    
                    # 处理重复表头
                    if base_header in header_count:
                        header_count[base_header] += 1
                        # 添加序号：建议送样量1、建议送样量2...
                        new_col = f"{base_header}{header_count[base_header]}"
                    else:
                        header_count[base_header] = 1
                        new_col = f"{base_header}{header_count[base_header]}"
                    
                    new_columns.append(new_col)
                
                # 更新DataFrame的列名
                if new_columns:
                    df.columns = new_columns
            
            # ==========================================
            # 🟢 自动生成并保存 Schema (此时已经是英文Key)
            # ==========================================
            extracted_schema = self._extract_schema_from_dfs(df_dict)
            schema_manager.save_schema(self.department, extracted_schema)
            self.logger.info(f"Schema 更新完毕 (Key已转英文)，共 {len(extracted_schema)} 个字段")

            all_columns = set()
            for df in df_dict.values():
                all_columns.update(df.columns.tolist())
            
            all_nodes = []
            for sheet, df in df_dict.items():
                nodes = await self._process_sheet(
                    df=df,
                    sheet_name=sheet,
                    filename=filename,
                    user_id=user_id
                )
                all_nodes.extend(nodes)
            
            return ProcessResult(
                success=True,
                chunks=all_nodes, 
                total_chunks=len(all_nodes),
                metadata={
                    "sheets_count": len(df_dict),
                    "total_rows": sum(len(df) for df in df_dict.values()),
                    "department": self.department,
                    "uploader": user_id,
                }
            )
            
        except Exception as e:
            self.logger.exception(f"Excel处理失败: {filename}")
            return ProcessResult(success=False, chunks=[], error_message=str(e))
    
    async def _process_sheet(self, df: pd.DataFrame, sheet_name: str, filename: str, user_id: str) -> List[TextNode]:
        nodes = []
        columns = df.columns.tolist()
        
        for idx, row in df.iterrows():
            if row.isna().all(): continue
            
            # 1. 构建文本 (结构化格式，方便人类阅读和检索)
            text_parts = []
            row_data = {} # 原始数据保留
            field_mapping = {}
            
            # 收集所有字段值 - 首先收集所有非NaN值到row_data
            for col in columns:
                value = row[col]
                if not pd.isna(value):
                    row_data[col] = value
            
            # 然后处理用于text和field_mapping的值
            for col in columns:
                value = row[col]
                if pd.isna(value): continue
                str_value = str(value).strip()
                # 不过滤任何值，包括空字符串和"None"字符串值
                
                # 文本部分用中文表头: "物种: 小鼠"
                text_parts.append(f"{col}: {str_value}")
                
                # 2. 构建字段映射，用于结构化文本生成
                # 特殊处理样本类型字段，根据表格名称区分
                if col == "样本类型":
                    # 根据表格名称区分样本类型字段
                    if "单细胞项目经验查询" in sheet_name:
                        milvus_key = "sample_type_exp"
                    elif "单细胞样本类型细分" in sheet_name:
                        milvus_key = "sample_type_prep"
                    else:
                        milvus_key = "sample_type_exp"
                    logger.info(f"样本类型字段映射: {col} -> {milvus_key} = '{str_value}' (表格: {sheet_name})")
                else:
                    milvus_key = self._normalize_column_name(col)
                field_mapping[milvus_key] = str_value
                # 调试日志：记录列名映射
                if milvus_key in ['sample_type_exp', 'sample_type_prep', 'sampling_temperature', 'cell_count', 'clustering_rate']:
                    logger.info(f"字段映射: {col} -> {milvus_key} = '{str_value}' (类型: {type(value).__name__})")
            
            # 2. 生成结构化文本
            structured_text_parts = []
            
            # 直接从row_data中构建字段映射，确保包含所有非NaN值
            direct_mapping = {}
            for col, value in row_data.items():
                str_value = str(value).strip()
                # 不过滤任何值，包括空字符串和"None"字符串值
                direct_mapping[col] = str_value
            
            # 从row_data中构建字段映射，确保包含所有非NaN值
            # 优先使用预定义映射，然后使用动态映射
            row_field_mapping = {}
            for col, value in row_data.items():
                str_value = str(value).strip()
                # 只过滤空字符串，不过滤"None"字符串值，因为我们需要在后面处理
                if not str_value: continue
                
                # 特殊处理样本类型字段，根据表格名称区分
                if col == "样本类型":
                    if "单细胞项目经验查询" in sheet_name:
                        milvus_key = "sample_type_exp"
                    elif "单细胞样本类型细分" in sheet_name:
                        milvus_key = "sample_type_prep"
                    else:
                        milvus_key = "sample_type_exp"
                else:
                    milvus_key = self._normalize_column_name(col)
                row_field_mapping[milvus_key] = str_value
            
            # 合并两个映射，优先使用field_mapping中的值
            combined_mapping = {**row_field_mapping, **field_mapping}
            
            # 实验平台
            if "platform_type" in combined_mapping:
                structured_text_parts.append(f"该实验记录了单细胞平台类型为\"{combined_mapping['platform_type']}\"的样本。")
            
            # 样本信息
            sample_info_parts = []
            # 根据用户提供的对应关系，将以下字段归类到样本信息
            if "species" in combined_mapping:
                sample_info_parts.append(f"物种为\"{combined_mapping['species']}\"")
            if "sample_type_exp" in combined_mapping:
                sample_info_parts.append(f"样本类型为\"{combined_mapping['sample_type_exp']}\"")
            elif "sample_type_prep" in combined_mapping:
                sample_info_parts.append(f"样本类型为\"{combined_mapping['sample_type_prep']}\"")
            if "sample_detailed_type" in combined_mapping:
                sample_info_parts.append(f"详细类型是\"{combined_mapping['sample_detailed_type']}\"")
            if "experiment_protocol" in combined_mapping:
                sample_info_parts.append(f"实验方案采用\"{combined_mapping['experiment_protocol']}\"")
            if "tissue_weight" in combined_mapping:
                sample_info_parts.append(f"组织重量为{combined_mapping['tissue_weight']}")
            if "tissue_weight_unit" in combined_mapping:
                sample_info_parts.append(f"{combined_mapping['tissue_weight_unit']}")
            if "qualitative_description" in combined_mapping:
                sample_info_parts.append(f"定性描述为\"{combined_mapping['qualitative_description']}\"")
            if "rin_score" in combined_mapping:
                sample_info_parts.append(f"核酸质量RIN值为{combined_mapping['rin_score']}")
            if "is_streaming" in combined_mapping:
                sample_info_parts.append(f"流式检测为{combined_mapping['is_streaming']}")
            if "antibody_info" in combined_mapping:
                sample_info_parts.append(f"抗体信息为\"{combined_mapping['antibody_info']}\"")
            if "streaming_protocol" in combined_mapping:
                sample_info_parts.append(f"流式方案为\"{combined_mapping['streaming_protocol']}\"")
            if "is_lysis" in combined_mapping:
                sample_info_parts.append(f"裂红处理为{combined_mapping['is_lysis']}")
            if "is_dead_removal" in combined_mapping:
                sample_info_parts.append(f"去死处理为{combined_mapping['is_dead_removal']}")
            
            # 直接从direct_mapping中获取样本信息字段
            for col, value in direct_mapping.items():
                if "物种" in col and not any("物种为" in part for part in sample_info_parts):
                    sample_info_parts.append(f"物种为\"{value}\"")
                elif "样本类型" in col and not any("样本类型为" in part for part in sample_info_parts):
                    sample_info_parts.append(f"样本类型为\"{value}\"")
                elif "样本详细类型" in col and not any("详细类型是" in part for part in sample_info_parts):
                    sample_info_parts.append(f"详细类型是\"{value}\"")
                elif "实验方案" in col and not any("实验方案采用" in part for part in sample_info_parts):
                    sample_info_parts.append(f"实验方案采用\"{value}\"")
                elif "组织重量" in col and not any("组织重量为" in part for part in sample_info_parts):
                    sample_info_parts.append(f"组织重量为{value}")
                elif "定性描述" in col and not any("定性描述为" in part for part in sample_info_parts):
                    sample_info_parts.append(f"定性描述为\"{value}\"")
                elif "核酸质量" in col or "RIN值" in col:
                    if not any("核酸质量RIN值为" in part for part in sample_info_parts):
                        sample_info_parts.append(f"核酸质量RIN值为{value}")
                elif "是否流式" in col and not any("流式检测为" in part for part in sample_info_parts):
                    sample_info_parts.append(f"流式检测为{value}")
                elif "抗体信息" in col and not any("抗体信息为" in part for part in sample_info_parts):
                    sample_info_parts.append(f"抗体信息为\"{value}\"")
                elif "流式方案" in col and not any("流式方案为" in part for part in sample_info_parts):
                    sample_info_parts.append(f"流式方案为\"{value}\"")
                elif "是否裂红" in col and not any("裂红处理为" in part for part in sample_info_parts):
                    sample_info_parts.append(f"裂红处理为{value}")
                elif "是否去死" in col and not any("去死处理为" in part for part in sample_info_parts):
                    sample_info_parts.append(f"去死处理为{value}")
            
            # 到样温度 - 支持多种字段名
            temp_key = None
            if "arrival_temp_celsius" in combined_mapping:
                temp_key = "arrival_temp_celsius"
            elif "sampling_temperature" in combined_mapping:
                temp_key = "sampling_temperature"
            if temp_key:
                # 检查是否已经包含温度单位
                temp_value = combined_mapping[temp_key]
                # 不过滤任何值，包括"None"字符串值
                if "℃" in temp_value or "°C" in temp_value:
                    sample_info_parts.append(f"到样温度为{temp_value}")
                else:
                    sample_info_parts.append(f"到样温度为{temp_value}℃")
            
            # 直接从direct_mapping中获取到样温度
            for col, value in direct_mapping.items():
                if "到样温度" in col:
                    temp_value = value
                    if "℃" in temp_value or "°C" in temp_value:
                        if not any("到样温度" in part for part in sample_info_parts):
                            sample_info_parts.append(f"到样温度为{temp_value}")
                    else:
                        if not any("到样温度" in part for part in sample_info_parts):
                            sample_info_parts.append(f"到样温度为{temp_value}℃")
            
            if sample_info_parts:
                structured_text_parts.append(f"样本信息：{', '.join(sample_info_parts)}。")
            
            # 实验指标
            exp_metrics_parts = []
            # 根据用户提供的对应关系，将以下字段归类到实验指标
            # 细胞总量 - 支持多种字段名
            cell_count_key = None
            if "total_cells_10k" in combined_mapping:
                cell_count_key = "total_cells_10k"
            elif "cell_count" in combined_mapping:
                cell_count_key = "cell_count"
            if cell_count_key:
                cell_count_value = combined_mapping[cell_count_key]
                # 不过滤任何值，包括"None"字符串值
                exp_metrics_parts.append(f"细胞总量{cell_count_value}万")
            
            # 直接从direct_mapping中获取细胞总量
            for col, value in direct_mapping.items():
                if "细胞总量" in col:
                    cell_count_value = value
                    if not any("细胞总量" in part for part in exp_metrics_parts):
                        exp_metrics_parts.append(f"细胞总量{cell_count_value}万")
            
            # 结团率 - 支持多种字段名
            clustering_rate_key = None
            if "clumping_rate_percent" in combined_mapping:
                clustering_rate_key = "clumping_rate_percent"
            elif "clustering_rate" in combined_mapping:
                clustering_rate_key = "clustering_rate"
            if clustering_rate_key:
                clustering_rate_value = combined_mapping[clustering_rate_key]
                # 不过滤任何值，包括"None"字符串值
                exp_metrics_parts.append(f"结团率{clustering_rate_value}%")
            
            # 直接从direct_mapping中获取结团率
            for col, value in direct_mapping.items():
                if "结团率" in col:
                    clustering_rate_value = value
                    if not any("结团率" in part for part in exp_metrics_parts):
                        exp_metrics_parts.append(f"结团率{clustering_rate_value}%")
            
            # 细胞活率
            if "cell_viability_percent" in combined_mapping:
                exp_metrics_parts.append(f"细胞活率{combined_mapping['cell_viability_percent']}%")
            # 直接从direct_mapping中获取细胞活率
            for col, value in direct_mapping.items():
                if "细胞活率" in col and not any("细胞活率" in part for part in exp_metrics_parts):
                    exp_metrics_parts.append(f"细胞活率{value}%")
            
            # 有核率
            if "nucleated_rate_percent" in combined_mapping:
                exp_metrics_parts.append(f"有核率{combined_mapping['nucleated_rate_percent']}%")
            # 直接从direct_mapping中获取有核率
            for col, value in direct_mapping.items():
                if "有核率" in col and not any("有核率" in part for part in exp_metrics_parts):
                    exp_metrics_parts.append(f"有核率{value}%")
            
            if exp_metrics_parts:
                structured_text_parts.append(f"实验指标：{', '.join(exp_metrics_parts)}。")
            
            # 数据指标
            data_metrics_parts = []
            # 根据用户提供的对应关系，将以下字段归类到数据指标
            # 捕获细胞数
            if "captured_cells" in combined_mapping:
                data_metrics_parts.append(f"捕获细胞数{combined_mapping['captured_cells']}")
            # 直接从direct_mapping中获取捕获细胞数
            for col, value in direct_mapping.items():
                if "捕获细胞数" in col and not any("捕获细胞数" in part for part in data_metrics_parts):
                    data_metrics_parts.append(f"捕获细胞数{value}")
            
            # 平均reads/cell
            if "reads_per_cell" in combined_mapping:
                data_metrics_parts.append(f"平均reads/cell为{combined_mapping['reads_per_cell']}")
            # 直接从direct_mapping中获取reads/cell
            for col, value in direct_mapping.items():
                if "reads/cell" in col and not any("平均reads/cell为" in part for part in data_metrics_parts):
                    data_metrics_parts.append(f"平均reads/cell为{value}")
            
            # 基因中位数
            if "median_genes" in combined_mapping:
                data_metrics_parts.append(f"基因中位数为{combined_mapping['median_genes']}")
            # 直接从direct_mapping中获取基因中位数
            for col, value in direct_mapping.items():
                if "基因中位数" in col and not any("基因中位数为" in part for part in data_metrics_parts):
                    data_metrics_parts.append(f"基因中位数为{value}")
            
            # 人工注释结果
            if "annotation_results" in combined_mapping:
                data_metrics_parts.append(f"人工细胞注释为：{combined_mapping['annotation_results']}")
            # 直接从direct_mapping中获取人工细胞注释
            for col, value in direct_mapping.items():
                if "人工细胞注释" in col and not any("人工细胞注释为" in part for part in data_metrics_parts):
                    data_metrics_parts.append(f"人工细胞注释为：{value}")
            
            if data_metrics_parts:
                structured_text_parts.append(f"数据指标：{', '.join(data_metrics_parts)}。")
            
            # 参考资料
            ref_parts = []
            # 根据用户提供的对应关系，将以下字段归类到参考资料
            if "digestion_protocol" in combined_mapping:
                ref_parts.append(f"组织消化方案为\"{combined_mapping['digestion_protocol']}\"")
            elif "tissue_digestion_protocol_name" in combined_mapping:
                ref_parts.append(f"组织消化方案为\"{combined_mapping['tissue_digestion_protocol_name']}\"")
            # 直接从direct_mapping中获取组织消化方案
            for col, value in direct_mapping.items():
                if "组织消化方案" in col and not any("组织消化方案为" in part for part in ref_parts):
                    ref_parts.append(f"组织消化方案为\"{value}\"")
            
            if "related_articles" in combined_mapping:
                ref_parts.append(f"相关文章链接为{combined_mapping['related_articles']}")
            elif "related_article_link" in combined_mapping:
                ref_parts.append(f"相关文章链接为{combined_mapping['related_article_link']}")
            # 直接从direct_mapping中获取相关文章链接
            for col, value in direct_mapping.items():
                if "相关组织用户文章链接" in col and not any("相关文章链接为" in part for part in ref_parts):
                    ref_parts.append(f"相关文章链接为{value}")
            
            if "feishu_doc_link" in combined_mapping:
                ref_parts.append(f"飞书文档链接为{combined_mapping['feishu_doc_link']}")
            # 直接从direct_mapping中获取飞书文档链接
            for col, value in direct_mapping.items():
                if "飞书文档链接" in col and not any("飞书文档链接为" in part for part in ref_parts):
                    ref_parts.append(f"飞书文档链接为{value}")
            
            if "video_live_link" in combined_mapping:
                ref_parts.append(f"视频直播链接为{combined_mapping['video_live_link']}")
            elif "video_stream_link" in combined_mapping:
                ref_parts.append(f"视频直播链接为{combined_mapping['video_stream_link']}")
            # 直接从direct_mapping中获取视频直播链接
            for col, value in direct_mapping.items():
                if "视频直播链接" in col and not any("视频直播链接为" in part for part in ref_parts):
                    ref_parts.append(f"视频直播链接为{value}")
            
            if ref_parts:
                structured_text_parts.append(f"参考资料：{', '.join(ref_parts)}。")
            
            # 生成最终文本
            if structured_text_parts:
                # 添加所有未在结构化文本中包含的字段
                # 收集已在结构化文本中包含的字段名
                included_fields = set()
                for part in structured_text_parts:
                    # 简单提取字段名（这只是一个近似方法，实际可能需要更复杂的解析）
                    if "：" in part:
                        field_part = part.split("：")[0]
                        included_fields.add(field_part)
                
                # 添加未包含的字段
                additional_parts = []
                for col, value in row_data.items():
                    str_value = str(value).strip()
                    if not str_value or str_value == "None":
                        continue
                    if col not in included_fields:
                        additional_parts.append(f"{col}: {str_value}")
                
                # 合并结构化文本和额外字段
                if additional_parts:
                    text_content = "\n".join(structured_text_parts + ["其他信息：" + "， ".join(additional_parts) + "。"])
                else:
                    text_content = "\n".join(structured_text_parts)
            else:
                # 回退到原始格式
                text_content = " | ".join(text_parts)
            
            # 2. 构建 Metadata (🔥 必须使用转换后的英文Key)
            clean_metadata = {
                "department": self.department,
                "uploader": user_id,
                "filename": filename,
                "doc_type": "excel",
                "sheet_name": sheet_name,
                "row_index": int(idx),
                "full_row_json": json.dumps(row_data, ensure_ascii=False),
            }

            # 遍历列，转换 Key 并存入 metadata
            for col in columns:
                if pd.isna(row[col]): continue
                
                milvus_value = str(row[col]).strip()
                # 不过滤任何值，包括空字符串和"None"字符串值
                
                # 特殊处理样本类型字段，根据表格名称区分
                if col == "样本类型":
                    # 根据表格名称区分样本类型字段
                    if "单细胞项目经验查询" in sheet_name:
                        milvus_key = "sample_type_exp"
                    elif "单细胞样本类型细分" in sheet_name:
                        milvus_key = "sample_type_prep"
                    else:
                        milvus_key = "sample_type_exp"
                    logger.info(f"样本类型字段映射 (metadata): {col} -> {milvus_key} = '{milvus_value}' (表格: {sheet_name})")
                else:
                    # 🔥 调用转换函数：中文 -> 英文
                    milvus_key = self._normalize_column_name(col)
                
                # 直接添加字段，因为我们已经在 COLUMN_MAPPING 中统一了映射关系
                clean_metadata[milvus_key] = milvus_value
            
            # 3. 创建 Node
            node = TextNode(
                text=text_content,
                id_=f"{self.department}_{uuid.uuid4().hex}",
                metadata=clean_metadata,
                excluded_llm_metadata_keys=["full_row_json"],
                excluded_embed_metadata_keys=["full_row_json"],
            )
            nodes.append(node)
        
        self.logger.info(f"✅ Sheet '{sheet_name}' -> {len(nodes)} nodes")
        return nodes
    
    # ==========================================
    # 🔥 2. 核心修复：列名标准化逻辑（支持自动拼音转换）
    # ==========================================
    def _normalize_column_name(self, column: str) -> str:
        """
        将Excel列名转换为 Milvus 安全的 Key。
        优先使用预定义映射表，然后使用动态映射表，否则自动转换为拼音。
        确保生成的字段名只包含字母、数字和下划线，符合Milvus查询要求。
        """
        # 0. 彻底清理所有特殊字符
        raw_col = str(column).strip()
        clean_col = raw_col
        
        # 1. 清理换行符、制表符等空白字符 - 这是关键修复！
        clean_col = re.sub(r'[\n\r\t]', '', clean_col)
        
        # 2. 清理中文括号、英文括号、冒号等特殊字符 - 这是关键修复！
        clean_col = clean_col.replace('（', '').replace('）', '')
        clean_col = clean_col.replace('(', '').replace(')', '')
        clean_col = clean_col.replace('：', '').replace(':', '')
        clean_col = clean_col.replace('、', '').replace('，', '')
        clean_col = clean_col.replace('。', '').replace('；', '')
        clean_col = clean_col.replace('“', '').replace('”', '')
        clean_col = clean_col.replace("'", '').replace('"', '')
        
        # 3. 清理斜杠等特殊字符，替换为下划线
        clean_col = clean_col.replace('/', '_').replace('\\', '_')
        clean_col = clean_col.replace('－', '_').replace('-', '_')
        
        # 4. 只保留字母、数字、中文和下划线
        clean_col = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff_]', '', clean_col)
        
        # 5. 优先使用预定义映射表 (Map "物种" -> "species")
        if clean_col in COLUMN_MAPPING:
            return COLUMN_MAPPING[clean_col]
        # 特殊处理 reads/cell 字段
        if raw_col == "reads/cell":
            return "reads_per_cell"
        
        # 6. 检查动态映射表，优先使用清理后的列名，避免特殊字符问题
        global DYNAMIC_COLUMN_MAPPING
        if clean_col in DYNAMIC_COLUMN_MAPPING:
            return DYNAMIC_COLUMN_MAPPING[clean_col]
        if raw_col in DYNAMIC_COLUMN_MAPPING:
            # 如果原始列名在映射表中，但包含特殊字符，更新映射表
            logger.warning(f"⚠️ 原始列名 '{raw_col}' 包含特殊字符，更新映射关系")
            # 删除旧的映射关系
            del DYNAMIC_COLUMN_MAPPING[raw_col]
            # 保存映射表
            self._save_dynamic_mapping()
            
        # 7. 自动生成拼音映射
        # 7.1 提取纯中文部分（去除前缀如"col_"）
        # 去除可能的前缀
        base_col = clean_col
        if base_col.startswith('col_'):
            base_col = base_col[4:]
        
        # 7.2 检查是否包含中文
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in base_col)
        
        if has_chinese:
            # 7.3 使用拼音转换
            # - 全拼风格用于完整拼音
            full_pinyin_parts = lazy_pinyin(base_col, style=Style.NORMAL, strict=False)
            
            # 7.4 生成安全的字段名
            # 方案：完整拼音，用下划线连接
            normalized = '_'.join(full_pinyin_parts).lower()
        else:
            # 8. 非中文列名处理
            # 替换非法字符为下划线
            normalized = ''.join(
                c if c.isalnum() or c == '_' else '_' 
                for c in clean_col
            )
            normalized = normalized.lower()
        
        # 9. 清理和标准化
        # 合并连续下划线
        while '__' in normalized:
            normalized = normalized.replace('__', '_')
        # 去除首尾下划线
        normalized = normalized.strip('_')
        
        # 10. 确保符合Milvus要求
        # Milvus不允许空字符串
        if not normalized:
            normalized = f"col_{uuid.uuid4().hex[:8]}"
        
        # 11. 添加前缀（可选，根据配置）
        # 只有当它不是标准英文词且未以配置的前缀开头时，才添加前缀
        if settings.EXCEL_COLUMN_PREFIX and not normalized.startswith(settings.EXCEL_COLUMN_PREFIX):
            # 检查是否为已有的标准英文词（简单判断：是否包含下划线）
            is_standard = '_' not in normalized
            if not is_standard:
                normalized = f"{settings.EXCEL_COLUMN_PREFIX}{normalized}"
        
        # 12. 更新动态映射表，同时保存原始列名和清理后的列名映射
        if raw_col not in COLUMN_MAPPING:
            if raw_col not in DYNAMIC_COLUMN_MAPPING or DYNAMIC_COLUMN_MAPPING[raw_col] != normalized:
                DYNAMIC_COLUMN_MAPPING[raw_col] = normalized
                logger.info(f"📝 更新动态映射: {raw_col} -> {normalized}")
                # 保存映射表
                self._save_dynamic_mapping()
        
        return normalized

    def _extract_schema_from_dfs(self, df_dict: Dict[str, pd.DataFrame]) -> List[Dict]:
        """提取 Schema 定义 (使用英文 Key)"""
        schema_map = {} 
        
        for sheet_name, df in df_dict.items():
            for col in df.columns:
                # 🔥 获取英文 Key
                norm_key = self._normalize_column_name(col)
                
                # 清理列名用于显示
                clean_col = str(col).strip()
                clean_col = re.sub(r'[\n\r\t]', '', clean_col)
                
                unique_values = df[col].dropna().unique()
                
                # 对于组织类型（tissue）字段，放宽唯一值数量限制
                if norm_key == "tissue":
                    # 组织类型可能有很多种，放宽限制到200个
                    is_enum = len(unique_values) <= 200 and len(unique_values) > 0
                    valid_values = [str(v) for v in unique_values[:200]] if is_enum else []
                else:
                    # 其他字段保持原有限制
                    is_enum = len(unique_values) <= 50 and len(unique_values) > 0
                    valid_values = [str(v) for v in unique_values[:50]] if is_enum else []
                
                if norm_key not in schema_map:
                    schema_map[norm_key] = {
                        "key": norm_key,       # species
                        "name": clean_col,      # 清理后的中文名给前端展示/LLM理解
                        "desc": f"来自表格列 '{clean_col}'",
                        "valid_values": valid_values
                    }
                else:
                    if is_enum:
                        existing = set(schema_map[norm_key]["valid_values"])
                        new_v = set(valid_values)
                        merged = list(existing | new_v)
                        if len(merged) <= 50:
                            schema_map[norm_key]["valid_values"] = merged

        return list(schema_map.values())