import json
import os
from typing import List, Dict, Any

# 存储在一个 JSON 文件中，生产环境建议存入 Redis 或 SQL 数据库
SCHEMA_FILE = "data/schema_registry.json"

class SchemaManager:
    @staticmethod
    def save_schema(department: str, columns_def: List[Dict]):
        """
        保存部门的数据结构定义
        """
        # 读取现有
        data = {}
        if os.path.exists(SCHEMA_FILE):
            try:
                with open(SCHEMA_FILE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except:
                pass
        
        # 更新该部门的 Schema
        # 这里做一个简单的合并策略：新的覆盖旧的
        data[department] = columns_def
        
        # 写入
        os.makedirs(os.path.dirname(SCHEMA_FILE), exist_ok=True)
        with open(SCHEMA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def get_schema(department: str) -> List[Dict]:
        """
        获取部门的 Schema 用于构建 LLM Prompt
        """
        if not os.path.exists(SCHEMA_FILE):
            return []
        
        try:
            with open(SCHEMA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get(department, [])
        except:
            return []

schema_manager = SchemaManager()