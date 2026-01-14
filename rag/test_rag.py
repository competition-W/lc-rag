"""
curl -X POST "http://127.0.0.1:8088/query" \
     -H "Content-Type: application/json" \
     -H "X-User-Id: 1" \
     -H "X-Department: market" \
     -H "Authorization: Bearer YOUR_TEST_TOKEN" \
     -d '{
           "text": "帮我筛选出物种是小鼠，且组织类型是心脏的样本",
           "use_llm": true
         }'
"""


import requests
import json
import time
from llama_index.core import Settings
from llama_index.llms.dashscope import DashScope
import os

# ================= 配置区 =================
API_URL = "http://127.0.0.1:8000/query"  # 您的 API 地址
TOKEN = "YOUR_ACCESS_TOKEN_HERE"         # 🔑 请替换为您的有效 JWT Token
# =========================================

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "X-User-Id": "1",
    "X-Department": "market"
}

def test_query(scenario_name, query_text):
    print(f"\n{'='*20} 测试场景: {scenario_name} {'='*20}")
    print(f"❓ 问题: {query_text}")
    
    payload = {
        "text": query_text,
        "use_llm": True
    }
    
    try:
        start_time = time.time()
        response = requests.post(API_URL, json=payload, headers=HEADERS)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            res_json = response.json()
            if res_json['code'] == 200: # 假设 utils.response.success 返回 code 200
                data = res_json['data']
                
                print(f"✅ 响应成功 ({duration:.2f}s)")
                print(f"🛠️  模式 (Mode): {data.get('mode')}")
                
                # 打印意图识别结果
                intent = data.get('intent', {})
                print(f"🧠 识别 Filters: {intent.get('extracted_filters')}")
                print(f"🔍 识别 Query: '{intent.get('search_term')}'")
                
                # 打印检索到的数据源数量
                sources = data.get('sources', [])
                print(f"📚 检索到数据条数: {len(sources)}")
                
                # 打印第一条数据的元数据（验证是否查对）
                if sources:
                    first_meta = sources[0].get('metadata', {})
                    # 过滤掉长字段方便显示
                    display_meta = {k:v for k,v in first_meta.items() if k != 'full_row_json'}
                    print(f"📄 第一条数据Metadata: {display_meta}")
                
                print(f"🤖 LLM 回答:\n{data.get('answer')}")
            else:
                print(f"❌ 业务逻辑错误: {res_json}")
        else:
            print(f"❌ HTTP 请求失败: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ 发生异常: {e}")

if __name__ == "__main__":


    # 查询测试
    test_query("语义查询查询", "小鼠心脏冻存组织样本的保存方案是怎样的？")
    
    # # 场景 1: 查表模式 (Schema 匹配测试)
    # # 预期: mode="table", filters={"col_wu_zhong": "猪"}
    # test_query("结构化查表", "帮我找一下猪的所有实验数据")
    
    # # 场景 2: 语义搜索模式 (非枚举列测试)
    # # 预期: mode="semantic", filters={} (假设表格里有 '结果' 列，但不是枚举，或者问的是描述性内容)
    # test_query("语义模糊搜索", "哪些样本的实验结果显示有炎症反应？")
    
    # # 场景 3: 混合/复杂模式
    # # 预期: mode="table", filters={"col_wu_zhong": "小鼠", "col_zu_zhi": "脑"}
    # test_query("多条件筛选", "查看小鼠大脑的相关记录")
    
    # # 场景 4: 幻觉测试 (不存在的值)
    # # 预期: filters={} (因为'霸王龙'不在枚举表里)，mode="semantic" 或 empty
    # test_query("防幻觉测试", "查找霸王龙的数据")