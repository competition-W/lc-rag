from pymilvus import connections, Collection

# 1. 配置信息
MILVUS_HOST = "110.1.122.1"
MILVUS_PORT = "30530"
COLLECTION_NAME = "dept_market" 

def simple_check():
    try:
        print(f"🔌 连接 Milvus: {MILVUS_HOST}:{MILVUS_PORT}...")
        connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
        
        if not ((COLLECTION_NAME)):
            print(f"❌ 集合 '{COLLECTION_NAME}' 不存在！")
            return

        col = Collection(COLLECTION_NAME)
        col.load()
        total = col.num_entities
        print(f"✅ 集合加载成功，当前共有数据: {total} 条")

        # 检查annotation_results字段是否存在
        print("\n🔍 检查annotation_results字段...")
        
        # 获取集合的schema
        schema = col.schema
        field_names = [field.name for field in schema.fields]
        print(f"  集合中的字段: {field_names}")
        
        if "annotation_results" in field_names:
            print("✅ annotation_results字段存在于Milvus集合中！")
        else:
            print("❌ annotation_results字段不存在于Milvus集合中！")
            return
        
        # 尝试查询有值的annotation_results记录
        print("\n🔍 尝试查询有值的annotation_results记录...")
        try:
            # 查询1条有值的记录
            results = col.query(expr="annotation_results != ''", limit=1, output_fields=["annotation_results"])
            if results:
                print(f"✅ 找到有值的annotation_results记录: {results[0]['annotation_results'][:100]}...")
            else:
                print("⚠️ 没有找到有值的annotation_results记录")
                # 尝试查询所有记录，看看是否有数据
                results = col.query(expr="", limit=5, output_fields=["annotation_results"])
                print("\n🔍 随机查询5条记录的annotation_results字段:")
                for i, res in enumerate(results):
                    val = res.get('annotation_results', '空')
                    print(f"  第{i+1}条: '{val}'")
        except Exception as e:
            print(f"❌ 查询失败: {e}")
            # 尝试简单查询，不使用expr
            results = col.query(expr="", limit=5, output_fields=["annotation_results"])
            print("\n🔍 随机查询5条记录的annotation_results字段:")
            for i, res in enumerate(results):
                val = res.get('annotation_results', '空')
                print(f"  第{i+1}条: '{val}'")
        
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    simple_check()
