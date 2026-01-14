from pymilvus import connections, utility, Collection

# 配置 (请保持与服务器一致)
MILVUS_HOST = "110.1.122.1"
MILVUS_PORT = "30530"

# 您认为应该存在的集合名
TARGET_COLLECTION = "dept_market"

def clean_and_verify():
    print(f"🔌 连接 Milvus: {MILVUS_HOST}:{MILVUS_PORT}...")
    try:
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    # 1. 列出所有集合
    all_collections = utility.list_collections()
    print(f"\n📋 当前 Milvus 中的所有集合: {all_collections}")
    
    if not all_collections:
        print("   (Milvus 是完全空的)")
        return

    # 2. 询问用户是否删除 TARGET_COLLECTION
    if TARGET_COLLECTION in all_collections:
        print(f"\n🔥 发现目标集合: '{TARGET_COLLECTION}'")
        choice = input(f"⚠️ 是否要彻底删除 '{TARGET_COLLECTION}' 并清空数据? (y/n): ")
        
        if choice.lower() == 'y':
            utility.drop_collection(TARGET_COLLECTION)
            print(f"✅ 集合 '{TARGET_COLLECTION}' 已删除！")
        else:
            print("🚫 操作取消，保留旧数据。")
    else:
        print(f"\n⚠️ 警告: 您想操作 '{TARGET_COLLECTION}'，但它不在列表中！")
        print("   请检查您的 config.py，确认入库代码到底写到哪个集合里去了。")

    # 3. 再次确认状态
    print("\n🔄 最终状态检查...")
    if utility.has_collection(TARGET_COLLECTION):
        col = Collection(TARGET_COLLECTION)
        print(f"   集合 '{TARGET_COLLECTION}' 依然存在，包含 {col.num_entities} 条数据。")
    else:
        print(f"   集合 '{TARGET_COLLECTION}' 目前不存在 (这是正常的，请现在运行入库脚本)。")

if __name__ == "__main__":
    clean_and_verify()