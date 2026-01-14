import sys
import os
import json
import asyncio
import time
from datetime import datetime

# ==========================================
# 1. 环境准备：确保能导入你的项目模块
# ==========================================
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

print(f"📂 工作目录: {current_dir}")

try:
    # 导入你的主程序文件 (假设文件名是 rag_fusion_0808.py)
    import chatapp.rag_fusion as rag_app
    # 导入语料配置
    from db_connection.name_persist import map_reduce
    print("✅ 成功导入 rag_fusion_0808 和 map_reduce 配置")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确认本脚本放在 rag_fusion_0808.py 同级目录，且 db_connection 文件夹存在。")
    sys.exit(1)

# ==========================================
# 2. 配置
# ==========================================
INPUT_FILE = "test_dataset.json"
OUTPUT_FILE = "prediction_results.json"

async def init_engine():
    """
    模拟 FastAPI 的 lifespan 启动过程，初始化 RAG 引擎
    """
    print("\n🔧 正在初始化 RAG 系统 (连接 Milvus, 加载模型)...")
    
    # 1. 连接 Milvus
    rag_app.connect_milvus_db()
    
    # 2. 构建检索器
    # 注意：rag_fusion_0808.py 里的 initialize_query_engine 依赖全局变量 retrievers
    # 所以我们必须手动设置 rag_app.retrievers
    print("   正在构建检索器 (这可能需要几秒钟)...")
    retrievers = rag_app.build_retrievers(map_reduce=map_reduce)
    
    if not retrievers:
        print("❌ 错误：没有构建出任何 retriever，请检查 Milvus 连接或 map_reduce 配置。")
        sys.exit(1)
        
    rag_app.retrievers = retrievers # <--- 关键：设置模块内的全局变量
    print(f"   ✅ 检索器构建完成: {len(retrievers)} 个")

    # 3. 初始化查询引擎
    print("   正在初始化 QueryEngine...")
    query_engine = rag_app.initialize_query_engine()
    
    if query_engine:
        print("✅ RAG 引擎初始化成功！Ready to go.")
    else:
        print("❌ RAG 引擎初始化失败。")
        sys.exit(1)
        
    return query_engine

async def main():
    # 1. 初始化
    query_engine = await init_engine()

    # 2. 读取题目
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到 {INPUT_FILE}，请先运行生成数据的脚本。")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    results = []
    print(f"\n🚀 开始批量预测，共有 {len(test_data)} 道题...\n")

    for i, item in enumerate(test_data):
        question = item["question"]
        ground_truth = item["ground_truth"]
        
        print(f"[{i+1}/{len(test_data)}] 问: {question}")
        
        try:
           # 1. 计时开始
            start_time = time.time()
            ttft = 0.0  # 初始化 TTFT
            
            # === 调用 RAG ===
            response = await query_engine.aquery(question)
            
            answer_text = ""
            
            # 处理流式响应
            if hasattr(response, "response_gen"):
                is_first_token = True
                async for chunk in response.response_gen:
                    # 捕捉第一个 token 的时间
                    if is_first_token:
                        ttft = time.time() - start_time
                        is_first_token = False
                    
                    answer_text += chunk
            else:
                # 非流式的情况，TTFT 等于总耗时
                answer_text = str(response)
                ttft = time.time() - start_time

            # 2. 计时结束
            total_time = time.time() - start_time
            
            # 如果是流式但没捕捉到（极快），兜底处理
            if ttft == 0.0: 
                ttft = total_time
            
            # 获取引用上下文
            contexts = []
            if hasattr(response, "source_nodes"):
                for node in response.source_nodes:
                    contexts.append(node.node.get_content())
            
            # ===============================
            
            # 打印更详细的性能指标
            # replace(chr(10), ' ') 是为了把换行符换成空格，保持日志整洁
            print(f"    答: {answer_text[:30].replace(chr(10), ' ')}...") 
            print(f"    ⚡ TTFT(首字): {ttft:.2f}s | ⏱️ 总耗时: {total_time:.2f}s | 📚 引用: {len(contexts)}条")

            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer_text,
                "contexts": contexts,
                "metrics": {  # 顺便把性能指标也存进结果里，以后分析用
                    "ttft": ttft,
                    "total_time": total_time
                }
            })
            
        except Exception as e:
            print(f"❌ 第 {i+1} 题运行出错: {e}")
            results.append({
                "question": question,
                "ground_truth": ground_truth,
                "answer": "Error generating response",
                "contexts": []
            })

    # 3. 保存结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n🎉 预测完成！结果已保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    # 运行异步主程序
    asyncio.run(main())