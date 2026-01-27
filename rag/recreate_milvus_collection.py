#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milvus集合重建脚本
用于删除并重新创建Milvus集合，使用更新后的Schema
"""
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from services.milvus_manager import milvus_manager
from config import settings, get_collection_name


def recreate_collection(department: str):
    """
    删除并重新创建指定部门的Milvus集合
    
    Args:
        department: 部门名称
    """
    try:
        # 获取集合名称
        collection_name = get_collection_name(department)
        print(f"📋 开始处理集合: {collection_name}")
        
        # 1. 连接Milvus
        print("🔗 连接Milvus...")
        milvus_manager._connect_milvus(alias="recreate_collection")
        
        # 2. 检查集合是否存在
        from pymilvus import utility
        if utility.has_collection(collection_name, using="recreate_collection"):
            # 3. 删除集合
            print(f"🗑️  删除现有集合: {collection_name}")
            utility.drop_collection(collection_name, using="recreate_collection")
            print(f"✅ 集合 {collection_name} 删除成功")
        else:
            print(f"ℹ️  集合 {collection_name} 不存在，将创建新集合")
        
        # 4. 重新创建集合（会自动使用最新的Schema）
        print("🔧 重新创建集合...")
        vector_store = milvus_manager.get_vector_store(department)
        print(f"✅ 集合 {collection_name} 重新创建成功")
        
        # 5. 断开连接
        milvus_manager._disconnect_milvus(alias="recreate_collection")
        print("✅ 操作完成！")
        
    except Exception as e:
        print(f"❌ 操作失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    return True


if __name__ == "__main__":
    # 默认为 "market" 部门
    department = sys.argv[1] if len(sys.argv) > 1 else "market"
    print(f"🚀 重建 {department} 部门的Milvus集合")
    success = recreate_collection(department)
    sys.exit(0 if success else 1)
