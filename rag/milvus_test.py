from pymilvus import connections, Collection
import json

# 1. 这里的配置请和您 config.py 里的一致
MILVUS_HOST = "110.1.122.1"
MILVUS_PORT = "30530"
# ⚠️ 必须确认这个名字！您可以在 main.py 或 config.py 里找到 MILVUS_COLLECTION
COLLECTION_NAME = "dept_market" 

def inspect_data():
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

        # 2. 查询前 3 条数据，查看字段结构
        print("\n🔍 正在读取前 3 条数据...")
        # output_fields=["*"] 会列出所有标量字段
        results = col.query(expr="", limit=3, output_fields=["*"])

        for i, res in enumerate(results):
            print(f"\n--- 第 {i+1} 条数据 ---")
            # 过滤掉 embedding 向量，只看 Metadata
            display = {k: v for k, v in res.items() if k != "embedding" and k != "text_embedding"}
            print(json.dumps(display, indent=4, ensure_ascii=False))

            # === 自动诊断 ===
            keys = display.keys()
            print("  [诊断结果]: ", end="")
            if "species" in keys:
                print("✅ 字段正常 (包含 species)")
            elif "col_物种" in keys:
                print("❌ 字段未更新 (仍为 col_物种)，请重新入库！")
            else:
                print("❓ 缺少物种字段")
            
            # 检查部门
            dept = display.get("department", "未找到")
            print(f"  [部门检查]: 数据存储的部门是 '{dept}' (您的查询请求头是 market)")

    except Exception as e:
        print(f"❌ 发生错误: {e}")

if __name__ == "__main__":
    inspect_data()