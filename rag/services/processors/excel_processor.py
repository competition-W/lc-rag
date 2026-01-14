#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import io
import json
import uuid
import re
from typing import List, Dict, Any, Optional
import pandas as pd

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
# 这是解决 Milvus 报错的关键。所有用于过滤的字段必须在此定义英文名。
# =========================================================
COLUMN_MAPPING = {
    # === 核心字段 ===
    "物种": "species",
    "col_物种": "species",
    
    "样本详细类型": "tissue",
    "组织": "tissue",
    "组织类型": "tissue",
    "col_样本详细类型": "tissue",
    
    "实验平台": "platform",
    "平台": "platform",
    "col_实验平台": "platform",
    
    "样本大类": "category",
    "col_样本大类": "category",

    # === 处理相关 ===
    "样本保存方案": "storage_method",
    "col_样本保存方案": "storage_method",
    
    "是否裂红": "is_lysis",
    "col_是否裂红": "is_lysis",
    
    "是否去死": "is_dead_removal",
    "col_是否去死": "is_dead_removal",
    
    "核酸质量": "rin_score",
    "col_核酸质量": "rin_score",
    
    # === 其他业务字段 ===
    "送样日期": "sample_date",
    "项目编号": "project_id"
}

class ExcelProcessor(BaseDocumentProcessor):
    """Excel处理器（LlamaIndex集成版 - 已修复Key编码问题）"""
    
    def __init__(self, department: str):
        super().__init__(department)
        self.skip_rows = settings.EXCEL_SKIP_HEADER_ROWS
    
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
            
            # 读取Excel
            if sheet_name:
                df_dict = {sheet_name: pd.read_excel(file_obj, sheet_name=sheet_name, skiprows=self.skip_rows)}
            else:
                df_dict = pd.read_excel(file_obj, sheet_name=None, skiprows=self.skip_rows)
            
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
            
            # 1. 构建文本 (保留中文表头，方便人类阅读)
            text_parts = []
            row_data = {} # 原始数据保留
            
            for col in columns:
                value = row[col]
                if pd.isna(value): continue
                str_value = str(value).strip()
                if not str_value: continue
                
                # 文本部分用中文表头: "物种: 小鼠"
                text_parts.append(f"{col}: {str_value}")
                row_data[col] = value
            
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
                
                # 🔥 调用转换函数：中文 -> 英文
                milvus_key = self._normalize_column_name(col)
                milvus_value = str(row[col]).strip()
                
                if milvus_value:
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
    # 🔥 2. 核心修复：列名标准化逻辑
    # ==========================================
    def _normalize_column_name(self, column: str) -> str:
        """
        将Excel列名转换为 Milvus 安全的 Key。
        优先使用映射表，否则进行清理。
        """
        clean_col = str(column).strip()
        
        # 1. 优先查表 (Map "物种" -> "species")
        if clean_col in COLUMN_MAPPING:
            return COLUMN_MAPPING[clean_col]
            
        # 2. 如果不在表里，保留原有前缀逻辑，但必须处理特殊字符
        # 如果包含中文且没在映射表里，这可能会在 Milvus 报错
        # 建议：只允许字母、数字、下划线
        
        # 如果是 "col_xxx" 格式且不在 map 里，去掉前缀再处理一下？
        # 这里为了稳妥，我们把所有非 ASCII 字符替换掉，或者只保留能处理的
        
        # 简单的降级处理：替换非法字符
        normalized = ''.join(
            c if c.isalnum() or c in ['_', ' '] else '_' 
            for c in clean_col
        )
        normalized = normalized.lower().replace(' ', '_')
        while '__' in normalized:
            normalized = normalized.replace('__', '_')
            
        # 加上前缀 (如果之前代码强依赖 col_ 前缀，可以保留，但建议只有未映射的加)
        # 映射过的已经是标准英文(如 species)，不需要加 col_
        
        # 只有当它不是标准英文词时，才加前缀防止冲突
        if not normalized.startswith(settings.EXCEL_COLUMN_PREFIX):
             return f"{settings.EXCEL_COLUMN_PREFIX}{normalized}"
             
        return normalized

    def _extract_schema_from_dfs(self, df_dict: Dict[str, pd.DataFrame]) -> List[Dict]:
        """提取 Schema 定义 (使用英文 Key)"""
        schema_map = {} 
        
        for sheet_name, df in df_dict.items():
            for col in df.columns:
                # 🔥 获取英文 Key
                norm_key = self._normalize_column_name(col)
                
                unique_values = df[col].dropna().unique()
                is_enum = len(unique_values) <= 50 and len(unique_values) > 0
                
                valid_values = []
                if is_enum:
                    valid_values = [str(v) for v in unique_values[:50]]
                
                if norm_key not in schema_map:
                    schema_map[norm_key] = {
                        "key": norm_key,       # species
                        "name": str(col),      # 物种 (保留中文名给前端展示/LLM理解)
                        "desc": f"来自表格列 '{col}'",
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