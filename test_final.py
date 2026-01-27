#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最终测试脚本，验证所有修复是否成功
"""
import sys
print(f"Python版本: {sys.version}")

# 测试1: 检查async_embedding模块导入
print("\n📋 测试1: 检查async_embedding模块导入")
try:
    from rag.llm.async_embedding import get_async_embed_model
    print("✅ 成功导入get_async_embed_model")
    client = get_async_embed_model()
    print(f"✅ 成功创建客户端")
    print(f"   API Key状态: {'已配置' if client.api_key else '未配置'}")
    print(f"   模型名称: {client.model_name}")
except Exception as e:
    print(f"❌ 导入失败: {e}")

# 测试2: 检查document_tasks_async模块导入
print("\n📋 测试2: 检查document_tasks_async模块导入")
try:
    from rag.tasks.document_tasks_async import process_document_task_async
    print("✅ 成功导入process_document_task_async")
except Exception as e:
    print(f"❌ 导入失败: {e}")

# 测试3: 检查document_tasks模块导入
print("\n📋 测试3: 检查document_tasks模块导入")
try:
    from rag.tasks.document_tasks import process_document_task
    print("✅ 成功导入process_document_task")
except Exception as e:
    print(f"❌ 导入失败: {e}")

print("\n🎉 所有测试完成！")