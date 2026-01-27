#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导入测试脚本
用于诊断应用导入错误
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("开始测试导入...")

# 测试1: 导入基本模块
try:
    import os
    import uuid
    import asyncio
    print("✅ 基本模块导入成功")
except Exception as e:
    print(f"❌ 基本模块导入失败: {e}")
    sys.exit(1)

# 测试2: 导入llama_index模块
try:
    from llama_index.core import Settings
    print("✅ llama_index模块导入成功")
except Exception as e:
    print(f"❌ llama_index模块导入失败: {e}")
    sys.exit(1)

# 测试3: 导入本地模块
try:
    from rag.utils.logger import logger
    print("✅ 本地logger模块导入成功")
except Exception as e:
    print(f"❌ 本地logger模块导入失败: {e}")
    sys.exit(1)

# 测试4: 导入配置模块
try:
    from rag.config import settings
    print("✅ 配置模块导入成功")
except Exception as e:
    print(f"❌ 配置模块导入失败: {e}")
    sys.exit(1)

# 测试5: 导入llm模块
try:
    from rag.llm.inference import DashScopeLLM
    from rag.llm.get_embedding import get_embed_model
    print("✅ llm模块导入成功")
except Exception as e:
    print(f"❌ llm模块导入失败: {e}")
    sys.exit(1)

# 测试6: 导入服务模块
try:
    from rag.services.intent_recognizer import intent_recognizer
    print("✅ intent_recognizer模块导入成功")
except Exception as e:
    print(f"❌ intent_recognizer模块导入失败: {e}")
    sys.exit(1)

try:
    from rag.services.query_parser import parse_user_query
    print("✅ query_parser模块导入成功")
except Exception as e:
    print(f"❌ query_parser模块导入失败: {e}")
    sys.exit(1)

try:
    from rag.services.query_service import unified_query_service
    print("✅ query_service模块导入成功")
except Exception as e:
    print(f"❌ query_service模块导入失败: {e}")
    sys.exit(1)

# 测试7: 导入API模块
try:
    from rag.api.query import router as query_router
    print("✅ api.query模块导入成功")
except Exception as e:
    print(f"❌ api.query模块导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    from rag.api.upload import router as documents_router
    print("✅ api.upload模块导入成功")
except Exception as e:
    print(f"❌ api.upload模块导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 测试8: 导入main模块
try:
    from rag import main
    print("✅ main模块导入成功")
except Exception as e:
    print(f"❌ main模块导入失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n✅ 所有模块导入成功！")
