import os
import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from langchain_community.chat_models import ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings

# ================= 配置区 =================
# 输入文件（刚才跑出来的结果）
INPUT_FILE = "prediction_results.json"
OUTPUT_EXCEL = "rag_evaluation_report.xlsx"

# ⚠️ 确保你的环境变量里有 DASHSCOPE_API_KEY
# 如果没有，请取消下面这行的注释并填入
os.environ["DASHSCOPE_API_KEY"] = "sk-b285c03d1bf0401a977132de909f89ed"
# ==========================================

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ 找不到 {INPUT_FILE}，请先运行 evaluate_rag.py 生成预测结果。")
        return

    print("📖 正在读取预测结果...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 1. 转换数据格式
    # Ragas 要求 Dataset 格式，且字段名为: ['question', 'answer', 'contexts', 'ground_truth']
    ragas_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }

    for item in data:
        ragas_data["question"].append(item["question"])
        ragas_data["answer"].append(item["answer"])
        ragas_data["contexts"].append(item["contexts"]) # 这是一个 list[str]
        ragas_data["ground_truth"].append(item["ground_truth"])

    dataset = Dataset.from_dict(ragas_data)

    # 2. 配置“阅卷老师” (Judge Model)
    # 我们使用 Qwen-Max 或 Qwen-Plus 来进行评分，因为它理解中文更好
    print("👨‍🏫 正在初始化阅卷模型 (Qwen-Max)...")
    
    # 评测用的 LLM
    judge_llm = ChatTongyi(model_name="qwen-max") 
    
    # 评测用的 Embedding (计算相关性需要)
    judge_embeddings = DashScopeEmbeddings(model="text-embedding-v4")

    # 3. 开始评分
    print("🚀 开始 Ragas 自动化评分 (这可能需要几分钟)...")
    
    # 定义我们要测的指标
    metrics = [
        context_precision,  # 检索精度：检索到的内容里有多少是有用的？
        context_recall,     # 检索召回：标准答案需要的信息，你检索到了吗？
        faithfulness,       # 忠实度：你的回答是否完全基于检索到的内容（有没有瞎编）？
        answer_relevancy,   # 答案相关性：你的回答是否切题？
    ]

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False # 遇到某一条报错不中断
    )

    # 4. 输出结果
    print("\n" + "="*40)
    print("📊 最终得分:")
    print("="*40)
    print(results)
    
    # 保存详细报告到 Excel
    df = results.to_pandas()
    df.to_excel(OUTPUT_EXCEL, index=False)
    print(f"\n✅ 详细报告已保存为: {OUTPUT_EXCEL}")
    print("你可以下载这个 Excel 文件查看每一道题的具体得分。")

if __name__ == "__main__":
    main()