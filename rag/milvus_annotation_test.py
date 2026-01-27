from pymilvus import connections, Collection
import json

# 1. 这里的配置请和您 config.py 里的一致
MILVUS_HOST = "110.1.122.1"
MILVUS_PORT = "30530"
# ⚠️ 必须确认这个名字！您可以在 main.py 或 config.py 里找到 MILVUS_COLLECTION
COLLECTION_NAME = "dept_market" 

# 简单的对象转换函数
def convert_object(obj):
    """转换对象为可序列化类型"""
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    elif hasattr(obj, 'tolist'):
        return obj.tolist()
    else:
        return str(obj)

def inspect_annotation_results():
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

        # 2. 查询前 5 条数据，只获取需要的字段
        print("\n🔍 正在读取前 5 条数据...")
        results = col.query(expr="", limit=5, output_fields=["species", "sample_type_exp", "sample_detailed_type", "annotation_results"])

        for i, res in enumerate(results):
            print(f"\n--- 第 {i+1} 条数据 ---")
            # 手动打印每个字段，避免JSON序列化问题
            print(f"  species: '{res.get('species', '未知')}'")
            print(f"  sample_type_exp: '{res.get('sample_type_exp', '未知')}'")
            print(f"  sample_detailed_type: '{res.get('sample_detailed_type', '未知')}'")
            print(f"  annotation_results: '{res.get('annotation_results', '空')}'")

            # === 自动诊断 ===
            print("  [诊断结果]: ", end="")
            has_annotation = res.get('annotation_results') and res.get('annotation_results') != ""
            has_species = res.get('species') and res.get('species') != "未知"
            has_sample_type = res.get('sample_type_exp') and res.get('sample_type_exp') != "未知"
            has_detailed_type = res.get('sample_detailed_type') and res.get('sample_detailed_type') != "未知"
            
            if has_annotation:
                print("✅ annotation_results 字段有值！")
            else:
                print("❌ annotation_results 字段为空")
            
            if has_species:
                print("✅ species 字段正常")
            if has_sample_type:
                print("✅ sample_type_exp 字段正常")
            if has_detailed_type:
                print("✅ sample_detailed_type 字段正常")

    except Exception as e:
        print(f"❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_annotation_results()
