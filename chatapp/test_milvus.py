from pymilvus import connections, utility, Collection

def test_milvus_connection_and_list_collections():
    try:
        # 连接到Milvus服务器
        print("正在连接到Milvus服务器...")
        connections.connect(
            alias="default",
            # host="110.1.131.161",  # 你的服务器IP,外网测试
            # port="30530"          # Milvus默认端口
            # host="my-release-milvus", #都在 k8s 中，直接使用服务名
            host="110.1.122.1", 
            port="30530"          # Milvus默认端口
            
        )
        print("✅ 成功连接到Milvus服务器!")
        
        # 测试连接状态
        print(f"当前连接: {connections.list_connections()}")
        
        # 获取所有collection列表
        print("\n正在获取所有collection...")
        collections = utility.list_collections()
        
        if collections:
            print(f"✅ 发现 {len(collections)} 个collection:")
            for i, collection_name in enumerate(collections, 1):
                print(f"{i}. {collection_name}")
                
                # 获取每个collection的详细信息
                try:
                    collection = Collection(collection_name)
                    print(f"   - 实体数量: {collection.num_entities}")
                    # 移除对 is_loaded 的访问
                    print(f"   - Schema: {collection.schema}")
                    print()
                except Exception as e:
                    print(f"   - 获取详情时出错: {str(e)}")
                    print()
        else:
            print("⚠️  没有发现任何collection")
            
    except Exception as e:
        print(f"❌ 连接失败: {str(e)}")
        print("请检查:")
        print("1. 服务器IP地址是否正确")
        print("2. 端口30530是否开放")
        print("3. Milvus服务是否正在运行")
        return False
    
    finally:
        # 断开连接
        try:
            connections.disconnect("default")
            print("已断开连接")
        except:
            pass
    
    return True

# 运行测试
if __name__ == "__main__":
    test_milvus_connection_and_list_collections()