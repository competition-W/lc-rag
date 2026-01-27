from pymilvus import connections, Collection, utility
import json

# Milvus连接配置
MILVUS_HOST = "110.1.122.1"
MILVUS_PORT = "30530"

# 1. 连接Milvus
print(f"🔌 连接 Milvus: {MILVUS_HOST}:{MILVUS_PORT}...")
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, alias="default")

# 2. 获取所有集合
print("\n📋 列出所有集合...")
collections = utility.list_collections(using="default")

# 3. 遍历所有集合，获取schema信息
milvus_schema_info = {
    "timestamp": "2026-01-23",
    "milvus_server": f"{MILVUS_HOST}:{MILVUS_PORT}",
    "collections": []
}

for collection_name in collections:
    print(f"\n🔍 正在检查集合: {collection_name}")
    
    # 获取集合对象
    col = Collection(collection_name)
    
    # 获取集合统计信息
    col.load()
    total_entities = col.num_entities
    
    # 获取schema信息
    schema = col.schema
    schema_info = {
        "name": collection_name,
        "exists": True,
        "stats": {
            "row_count": total_entities
        },
        "schema": {
            "description": schema.description,
            "auto_id": schema.auto_id,
            "fields": []
        },
        "indexes": [],
        "sample_data": []
    }
    
    # 获取字段信息
    for field in schema.fields:
        field_info = {
            "name": field.name,
            "type": field.dtype.name,
            "is_primary": field.is_primary,
            "description": field.description,
            "dim": field.dim if hasattr(field, 'dim') else None,
            "max_length": field.max_length if hasattr(field, 'max_length') else None
        }
        schema_info["schema"]["fields"].append(field_info)
    
    # 获取索引信息
    indexes = col.indexes
    for idx in indexes:
        index_info = {
            "field_name": idx.field_name,
            "index_type": idx.params.get("index_type", "N/A"),
            "params": idx.params
        }
        schema_info["indexes"].append(index_info)
    
    # 获取样本数据
    if total_entities > 0:
        results = col.query(expr="", limit=1, output_fields=[field.name for field in schema.fields if field.dtype.name != 'FLOAT_VECTOR'])
        if results:
            sample_data = list(results[0].keys())
            schema_info["sample_data"] = sample_data
    
    # 添加到总信息中
    milvus_schema_info["collections"].append(schema_info)

# 4. 保存为JSON文件
output_file = "milvus_real_schema.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(milvus_schema_info, f, ensure_ascii=False, indent=4)

print(f"\n✅ Schema信息已保存到: {output_file}")
print(f"📊 共检查了 {len(collections)} 个集合")

# 5. 断开连接
connections.disconnect("default")

# 6. 打印schema字段详细信息
print("\n📋 打印第一个集合的schema.fields详细信息:")
if milvus_schema_info["collections"]:
    first_collection = milvus_schema_info["collections"][0]
    print(f"\n集合: {first_collection['name']}")
    print("字段列表:")
    for field in first_collection["schema"]["fields"]:
        print(f"- 名称: '{field['name']}', 类型: {field['type']}, 主键: {field['is_primary']}")

print("\n🔚 脚本执行完成!")