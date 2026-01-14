#!/mnt/chatchat/.venv/bin python3
# -*- coding: utf-8 -*-
def make_mesage(ground_truth_answer, prediction, query):
    '''请输入数据对哦'''
    system = ("用户提供三段输入，prediction，代表模型预测信息。\nground_truth_answer，代表需要参考的正确回答。\nquery，代表用户原始的咨询问题。\n请你判别prediction是否可接受。\n")
    instruction = (
    "上下文信息如下:\n"
    "---------------------\n"
    f"ground_truth_answer: {ground_truth_answer}\n"
    "---------------------\n"
    f"prediction: {prediction}\n"
    "---------------------\n"
    f"query: {query}\n"
    "---------------------\n\n\n"
    "请你根据上下文信息，判断prediction是否可接受。判别规则如下\n"
    "1.首先确认prediction是否符合正确回答的内容ground_truth_answer。\n"
    "2.期望模型回答(prediction)不要偏离用户问题(query)，保持针对用户问题的自我一致性\n"
    "3.输出一个[0,1]之间的数字，记为回答文本可信程度,越大可接受程度越高;反之模型回答越不可靠。\n"
    "4.你的回答模板（format）如下\n"
    "<answer> 输出文本可信程度</answer>\n"
    "<think> 输出简要的（concise）思考内容</think>\n"
    )
    message = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Instruction: {instruction}"}
    ]
    return message
import os
##########阿里百炼
os.environ["DASHSCOPE_API_KEY"] = "sk-eaa6c78b4d22459a8858b97d2dcac34e"
os.environ["DASHSCOPE_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from openai import OpenAI


def process_verify(ground_truth_answer: list, prediction: list, query:list):
    '''请输入数据对哦
    
    verifier的基座模型一定要强'''

    client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    verification = []
    for ans, pred, question in zip(ground_truth_answer, prediction, query):
        message = make_mesage(ans, pred, question)
        print("**************************************")
        print(message)
        completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        model="qwen-max-latest",
        messages=message,
        max_tokens=256
        # Qwen3模型通过enable_thinking参数控制思考过程（开源版默认True，商业版默认False）
        # 使用Qwen3开源版模型时，若未启用流式输出，请将下行取消注释，否则会报错
        # extra_body={"enable_thinking": False},
        )
        verification.append(completion.choices[0].message.content.strip())
        print(verification[-1])
    return verification


# system = ("用户提供两段输入，prediction和ground_truth_answer，请你判别prediction是否可接受")

# def make_mesage(ground_truth_answer, prediction):
#     '''请输入数据对哦'''
#     system = ("用户提供两段输入，prediction和ground_truth_answer，请你判别prediction是否可接受")
#     instruction = (
#     "上下文信息如下:\n"
#     "---------------------\n"
#     f"ground_truth_answer: {ground_truth_answer}\n"
#     "---------------------\n"
#     f"prediction: {prediction}\n"
#     "仅使用上下文信息，不需要先验知识，判别prediction是否可接受\n"
#     "输出一个[0,1]之间的数字,越大可接受程度越高。\n"
#     "Answer: "
#     )
#     message = [
#         {"role": "system", "content": system},
#         {"role": "user", "content": f"Instruction: {instruction}"}
#     ]
#     return message

# from llama_index.core.prompts import RichPromptTemplate

# from llama_index.core import PromptTemplate

# qa_tmpl_str = (
#     "Context information is below.\n"
#     "---------------------\n"
#     "{context_str}\n"
#     "---------------------\n"
#     "Given the context information and not prior knowledge, "
#     "answer the query.\n"
#     "Query: {query_str}\n"
#     "Answer: "
# )
# qa_tmpl = PromptTemplate(qa_tmpl_str)

# if __name__ == "__main__":
    # llm = DashScopeLLM(model_name="qwen-plus", temperature=0.8) ## 默认0.7; 0.8 top_p
    # llm.complete("请为'Fate 今晚留下来'写首一个字的诗。人物:saber,远坂凛，间桐樱，archer，卫宫士郎，伊莉雅。情节：圣杯战争。结果：正义的伙伴。").text