from pymilvus import connections, Collection

# 1. 这里的配置请和您 config.py 里的一致
MILVUS_HOST = "110.1.122.1"
MILVUS_PORT = "30530"
# ⚠️ 必须确认这个名字！您可以在 main.py 或 config.py 里找到 MILVUS_COLLECTION
COLLECTION_NAME = "dept_market" 

def check_annotation_results():
    try:
        print(f"🔌 连接 Milvus: {MILVUS_HOST}:{MILVUS_PORT}...")
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        
        if not ((COLLECTION_NAME)):
            print(f"❌ 集合 '{COLLECTION_NAME}' 不存在！请检查集合名称配置。")
            return

        col = Collection(COLLECTION_NAME)
        col.load()
        total = col.num_entities
        print(f"✅ 集合加载成功，当前共有数据: {total} 条")

        if total == 0:
            print("⚠️ 集合是空的！请运行入库脚本。")
            return

        # 2. 尝试查询包含注释结果的记录
        print("\n🔍 正在尝试查询包含注释结果的记录...")
        
        # 方法1: 查询10条最新的记录，看看其中是否有注释结果
        print("\n方法1: 查询10条最新的记录...")
        results = col.query(expr="", limit=10, output_fields=["id", "annotation_results", "text"])
        
        found = 0
        for i, res in enumerate(results):
            print(f"\n--- 第 {i+1} 条记录 ---")
            print(f"  ID: {res.get('id')}")
            print(f"  Annotation Results: '{res.get('annotation_results', '空')}'")
            
            # 检查text字段中是否包含注释信息
            text = res.get('text', '')
            if '人工注释结果' in text:
                print(f"  Text包含注释信息: {text[:100]}...")
            
            if res.get('annotation_results') and res.get('annotation_results') != "":
                found += 1
                print(f"  ✅ 找到有值的annotation_results！")
        
        print(f"\n在10条记录中找到 {found} 条包含注释结果的记录")
        
        # 方法2: 尝试搜索text字段中包含注释信息的记录
        print("\n方法2: 搜索text字段中包含注释信息的记录...")
        try:
            # 使用text字段搜索包含"人工注释结果"的记录
            results = col.search(
                data=[[0.0]*1024],  # 空向量，因为我们只关心text字段搜索
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=5,
                expr="text like '%人工注释结果%'",
                output_fields=["annotation_results", "text"]
            )
            
            if results:
                hits = results[0]
                print(f"  找到 {len(hits)} 条包含注释信息的记录")
                for i, hit in enumerate(hits):
                    print(f"\n  --- 搜索结果 {i+1} ---")
                    print(f"    Annotation Results: '{hit.entity.get('annotation_results', '空')}'")
                    print(f"    Text: {hit.entity.get('text', '')[:100]}...")
        except Exception as e:
            print(f"  搜索失败: {e}")
        
        # 方法3: 尝试搜索annotation_results字段中包含具体注释格式的记录
        print("\n方法3: 搜索annotation_results字段中包含具体注释格式的记录...")
        try:
            # 使用正则表达式搜索包含注释格式的记录
            results = col.search(
                data=[[0.0]*1024],  # 空向量，因为我们只关心字段搜索
                anns_field="embedding",
                param={"metric_type": "COSINE", "params": {"nprobe": 10}},
                limit=5,
                expr="annotation_results like '%Endothelial%'",
                output_fields=["annotation_results", "text"]
            )
            
            if results:
                hits = results[0]
                print(f"  找到 {len(hits)} 条包含Endothelial的记录")
                for i, hit in enumerate(hits):
                    print(f"\n  --- 搜索结果 {i+1} ---")
                    print(f"    Annotation Results: '{hit.entity.get('annotation_results', '空')}'")
                    print(f"    Text: {hit.entity.get('text', '')[:100]}...")
        except Exception as e:
            print(f"  搜索失败: {e}")
            print(f"  错误详情: {str(e)}")

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_annotation_results()
