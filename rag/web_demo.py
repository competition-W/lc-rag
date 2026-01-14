# 文件路径: /mnt/omicshub/rag/web_demo.py
import streamlit as st
import requests
import pandas as pd
import json
import time

# ================= 配置区域 =================
# 后端 API 地址 (根据你的实际情况修改 IP 和端口)
API_URL = "http://127.0.0.1:8088/query"
# 如果有鉴权，这里填入临时的 Bearer Token，或者在页面侧边栏输入
DEFAULT_TOKEN = "your_temp_token_here"
DEFAULT_USER_ID = "1"  # 假设用户 ID 为 1，如果有其他来源，可以在这里修改

# ================= 页面设置 =================
st.set_page_config(
    page_title="OmicsHub 智能检索",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 OmicsHub RAG 智能检索平台")
st.markdown("---")

# ================= 侧边栏：设置 =================
with st.sidebar:
    st.header("⚙️ 调试设置")
    api_url = st.text_input("API URL", value=API_URL)
    auth_token = st.text_input("Authorization Token", value=DEFAULT_TOKEN, type="password")
    user_id = st.text_input("User ID", value=DEFAULT_USER_ID)  # 这里去掉了 type="text"
    use_llm = st.checkbox("开启 LLM 回答", value=True)
    
    st.info("💡 提示：关闭 LLM 可以仅测试检索命中率，响应更快。")

# ================= 主逻辑 =================

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if prompt := st.chat_input("请输入问题，例如：帮我找小鼠心脏的数据..."):
    # 1. 显示用户问题
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. 调用后端 API
    with st.chat_message("assistant"):
        status_placeholder = st.empty()
        status_placeholder.markdown("🔍 **正在分析意图并检索数据...**")
        
        start_time = time.time()
        
        try:
            headers = {
                "Content-Type": "application/json",
                "X-User-Id": user_id,  # 添加用户 ID
                "X-Department": "market",  # 根据需要修改
                "Authorization": f"Bearer {auth_token}"  # 添加 Bearer Token
            }
            
            payload = {
                "text": prompt,
                "use_llm": use_llm
            }

            # 发送请求
            response = requests.post(api_url, json=payload, headers=headers)
            end_time = time.time()
            time_cost = end_time - start_time

            if response.status_code == 200:
                res_json = response.json()
                
                # 提取核心数据
                if res_json.get("code") == 200:
                    data = res_json.get("data", {})
                    answer = data.get("answer", "未生成回答")
                    intent = data.get("intent", {})
                    sources = data.get("sources", [])
                    all_rows = data.get("all_rows", [])
                    
                    status_placeholder.empty()  # 清除"正在加载"
                    
                    # ----------------- A. 展示 LLM 回答 -----------------
                    st.success(f"✅ 耗时: {time_cost:.2f}s | 命中数据: {len(all_rows)} 条")
                    if use_llm:
                        st.markdown("### 🤖 AI 回答")
                        st.markdown(answer)
                        # 将回答加入历史
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    else:
                        st.markdown("*LLM 生成已关闭，仅展示检索结果*")

                    # ----------------- B. 可视化调试面板 (Tabs) -----------------
                    tab1, tab2, tab3 = st.tabs(["🧠 意图解析 (Debug)", "📄 引用来源 (Sources)", "📊 完整数据表"])
                    
                    with tab1:
                        st.markdown("#### 意图识别结果")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("**提取的过滤器 (Filters):**")
                            st.json(intent.get("extracted_filters"))
                        with col2:
                            st.markdown("**最终搜索词 (Search Term):**")
                            term = intent.get("search_term")
                            if term == prompt:
                                st.code(term, language="text")
                                st.caption("⚠️ 提取词为空，回退使用原始提问")
                            else:
                                st.code(term, language="text")
                                st.caption("✅ 使用了提取后的精准关键词")

                    with tab2:
                        st.markdown(f"#### Top-{len(sources)} 高相关片段")
                        for i, src in enumerate(sources):
                            with st.expander(f"来源 #{i+1} (Score: {src.get('score', 0):.4f})"):
                                meta = src.get("metadata", {})
                                st.markdown(f"**文件**: `{meta.get('filename', 'N/A')}` | **物种**: `{meta.get('species', 'N/A')}`")
                                st.text(src.get("text", ""))

                    with tab3:
                        st.markdown(f"#### 检索到的所有数据 ({len(all_rows)} 条)")
                        if all_rows:
                            # 转换为 DataFrame 展示
                            flat_rows = []
                            for row in all_rows:
                                r_data = row.get("metadata", row) if isinstance(row, dict) else row
                                flat_rows.append(r_data)
                                
                            df = pd.DataFrame(flat_rows)
                            st.dataframe(df, use_container_width=True)
                        else:
                            st.warning("未检索到任何数据")

                else:
                    st.error(f"业务错误: {res_json.get('message')}")
            else:
                st.error(f"HTTP 请求失败: {response.status_code} - {response.text}")

        except Exception as e:
            st.error(f"发生异常: {str(e)}")