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