import os
import json
import asyncio
import random
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.dashscope import DashScope

# ================= 配置区域 =================

# 1. 你的阿里云 API Key
# ⚠️ 如果环境变量没设，请把 "sk-xxx" 换成你真实的 Key
API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-b285c03d1bf0401a977132de909f89ed")

# 2. 【已修改】原始文档存放的目录 (绝对路径)
SOURCE_DIR = "/mnt/ai/corpus_manager/test_data"

# 3. 输出文件路径 (生成在当前目录下)
OUTPUT_FILE = "test_dataset.json"

# 4. 生成多少对问答？
NUM_QUESTIONS = 20

# ===========================================

async def generate_dataset():
    print(f"📂 正在检查数据目录: {SOURCE_DIR}")
    
    # 1. 检查数据目录
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ 错误：找不到路径 {SOURCE_DIR}")
        print("请检查挂载是否正确，或者路径是否存在拼写错误。")
        return

    # 2. 读取文档
    try:
        # recursive=True 表示也会读取子文件夹里的文件
        reader = SimpleDirectoryReader(SOURCE_DIR, recursive=True)
        documents = reader.load_data()
        print(f"✅ 成功加载 {len(documents)} 个文档片段")
    except Exception as e:
        print(f"❌ 读取文档失败: {e}")
        return

    if len(documents) == 0:
        print("⚠️ 目录是存在的，但是没有读到任何文件。请检查里面是否有 PDF/TXT/Word 文件。")
        return

    # 3. 切分文档
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=100)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"📄 文档已切分为 {len(nodes)} 个节点")

    # 4. 初始化大模型
    llm = DashScope(model_name="qwen-max", api_key=API_KEY)

    # 5. 随机抽样
    selected_nodes = random.sample(nodes, min(NUM_QUESTIONS, len(nodes)))
    
    qa_dataset = []
    print(f"\n🚀 开始利用 AI 生成 {len(selected_nodes)} 组问答对...\n")

    for i, node in enumerate(selected_nodes):
        context = node.get_content().strip()
        if not context: continue

        prompt = (
            "你是一个专业的考试出题专家。请根据下面的[参考文本]，编写一个用户可能会问的[问题]以及对应的[标准答案]。\n"
            "要求：\n"
            "1. 问题必须能从文本中找到答案。\n"
            "2. 答案必须准确、简洁，且完全基于参考文本。\n"
            "3. 输出格式必须严格为 JSON 格式，包含 'question' 和 'ground_truth' 两个字段。\n\n"
            "[参考文本]：\n"
            f"{context[:2000]}\n\n"
            "JSON输出："
        )

        try:
            response = await llm.acomplete(prompt)
            raw_text = response.text.strip()
            
            if raw_text.startswith("```json"):  
                raw_text = raw_text[7:]  
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]
            
            qa_pair = json.loads(raw_text)
            question = qa_pair.get("question")
            ground_truth = qa_pair.get("ground_truth")

            if question and ground_truth:
                qa_dataset.append({
                    "question": question,
                    "ground_truth": ground_truth,
                    "source_context": context[:200] + "..." # 记录一部分原文方便定位
                })
                print(f"✅ [{i+1}/{len(selected_nodes)}] 生成成功: {question}")
            else:
                print(f"⚠️ [{i+1}] 格式错误")

        except Exception as e:
            print(f"❌ [{i+1}] 生成出错: {e}")

    # 6. 保存
    if qa_dataset:
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(qa_dataset, f, ensure_ascii=False, indent=2)
        print(f"\n🎉 完成！已生成文件: {OUTPUT_FILE}")
    else:
        print("\n😭 并没有生成任何数据。")

if __name__ == "__main__":
    asyncio.run(generate_dataset())