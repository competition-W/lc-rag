# 文件路径: /mnt/omicshub/rag/web_demo.py
import streamlit as st
import pandas as pd
import json
import time
import asyncio
import websockets
import httpx
import requests

# 引入日志配置
from utils.logger import logger

# ================= 配置区域 =================
# 后端 API 地址 (根据你的实际情况修改 IP 和端口)
API_URL = "http://127.0.0.1:8090/query"
# WebSocket地址
WS_URL = "ws://127.0.0.1:8090/query/ws"
# 如果有鉴权，这里填入临时的 Bearer Token，或者在页面侧边栏输入
DEFAULT_TOKEN = "your_temp_token_here"
DEFAULT_USER_ID = "1"  # 假设用户 ID 为 1，如果有其他来源，可以在这里修改

# 初始化WebSocket相关的session_state
if "connection_id" not in st.session_state:
    st.session_state.connection_id = None

if "current_answer" not in st.session_state:
    st.session_state.current_answer = ""

# ================= WebSocket管理 =================
async def handle_websocket_messages(ws, answer_placeholder):
    """处理WebSocket消息"""
    answer = ""
    try:
        while True:
            message = await ws.recv()
            data = json.loads(message)
            if data.get("type") == "content":
                content = data.get("value", "")
                answer += content
                # 更新回答占位符，这是唯一的显示位置
                answer_placeholder.markdown(f"{answer}")
            elif data.get("type") == "end_of_stream":
                # 流式结束，只更新session_state，不直接修改聊天历史
                st.session_state.current_answer = answer
                break
    except websockets.exceptions.ConnectionClosed as e:
        # 忽略正常关闭的连接错误，这通常发生在流式传输完成后
        if "sent 1000 (OK)" in str(e) and "received 1000 (OK)" in str(e):
            # 正常关闭，不显示错误
            pass
        else:
            st.error(f"❌ WebSocket连接意外关闭: {str(e)}")
    except Exception as e:
        st.error(f"❌ WebSocket消息处理失败: {str(e)}")
    return answer

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
    use_streaming = st.checkbox("开启流式输出", value=True)
    
    st.info("💡 提示：关闭 LLM 可以仅测试检索命中率，响应更快。")
    st.info("💡 流式输出：实时显示LLM生成的内容，提供更流畅的体验。")

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
        
        # 根据条件选择处理模式
        if use_streaming and use_llm:
            # 流式输出模式
            status_placeholder.markdown("🔄 **正在建立WebSocket连接...**")
            
            status_placeholder.markdown("📡 **正在生成流式回答...**")
            
            # 使用session_state来存储最终回答，避免nonlocal作用域问题
            st.session_state.final_answer = ""
            
            # 发送流式请求并处理响应
            async def stream_task():
                # 在同一个事件循环中创建WebSocket连接
                ws = None
                connection_id = None
                try:
                    logger.info(f"📞 开始流式请求处理，查询: '{prompt}'")
                    
                    # 1. 建立WebSocket连接
                    logger.info(f"🔌 尝试连接WebSocket服务器: {WS_URL}")
                    ws = await websockets.connect(WS_URL)
                    logger.info(f"✅ WebSocket连接成功")
                    
                    # 2. 接收连接信息
                    message = await ws.recv()
                    data = json.loads(message)
                    logger.info(f"📥 收到WebSocket连接信息: {data}")
                    if data.get("type") == "connection_info":
                        connection_id = data.get("connection_id")
                        st.session_state.connection_id = connection_id
                        logger.info(f"📋 获得WebSocket连接ID: {connection_id}")
                    else:
                        st.error(f"❌ 无法获取WebSocket连接信息: {data}")
                        logger.error(f"❌ 无法获取WebSocket连接信息: {data}")
                        return
                    
                    if not connection_id:
                        raise Exception("WebSocket连接失败，无法获取连接ID")
                    
                    # 3. 并行处理：同时发送HTTP请求和监听WebSocket消息
                    headers = {
                        "Content-Type": "application/json",
                        "X-User-Id": user_id,
                        "X-Department": "market",
                        "Authorization": f"Bearer {auth_token}"
                    }
                    
                    streaming_url = f"{api_url}/streaming"
                    payload = {
                        "text": prompt,
                        "connection_id": connection_id,
                        "use_llm": use_llm
                    }
                    
                    logger.info(f"📤 准备发送流式HTTP请求到: {streaming_url}")
                    logger.info(f"📋 请求参数: {payload}")
                    
                    # 创建一个空的占位符来显示LLM回答
                    answer_placeholder = st.empty()
                    
                    # 4. 并行执行：
                    #    a. 发送HTTP请求
                    #    b. 监听WebSocket消息（立即开始，不需要等待HTTP响应）
                    
                    # 创建WebSocket消息处理任务
                    ws_task = asyncio.create_task(handle_websocket_messages(ws, answer_placeholder))
                    logger.info(f"🔄 创建WebSocket消息处理任务")
                    
                    # 初始化变量
                    response = None
                    res_json = {"code": 200, "data": {}}
                    
                    # 发送HTTP请求，增加超时时间到5分钟
                    try:
                        async with httpx.AsyncClient(timeout=300.0) as client:
                            response = await client.post(streaming_url, json=payload, headers=headers)
                        end_time = time.time()
                        time_cost = end_time - start_time
                        logger.info(f"✅ HTTP请求成功，状态码: {response.status_code}，耗时: {time_cost:.2f}s")
                        
                        if response.status_code == 200:
                            res_json = response.json()
                            logger.info(f"📥 HTTP响应数据: {res_json.keys()}")
                            
                            # 检查all_rows的结构，确保不包含LLM回答
                            data = res_json.get("data", {})
                            if data is None:
                                data = {}
                            all_rows = data.get("all_rows", [])
                            logger.info(f"📋 all_rows类型: {type(all_rows)}, 长度: {len(all_rows)}")
                            if all_rows:
                                first_row = all_rows[0]
                                logger.info(f"🔍 第一个all_row的类型: {type(first_row)}, 包含键: {first_row.keys() if isinstance(first_row, dict) else '非字典类型'}")
                                
                            # 等待WebSocket消息处理完成
                            await ws_task
                            # 获取最终回答
                            st.session_state.final_answer = st.session_state.current_answer
                            logger.info(f"✅ WebSocket消息处理完成，最终回答长度: {len(st.session_state.final_answer)} 字符")
                            logger.info(f"📋 当前会话状态中的回答: {st.session_state.current_answer[:100]}...")
                            logger.info(f"📋 当前会话状态中的final_answer: {st.session_state.final_answer[:100]}...")
                            logger.info(f"📋 聊天历史最后一条: {st.session_state.messages[-1] if st.session_state.messages else '空'}")
                            
                            # 将最终回答添加到聊天历史，这是唯一的添加点
                            # 这样聊天历史中只会有完整的回答，不会重复显示
                            if st.session_state.final_answer:
                                logger.info(f"📝 将最终回答添加到聊天历史，避免重复显示")
                                st.session_state.messages.append({"role": "assistant", "content": st.session_state.final_answer})
                    except httpx.ReadTimeout:
                        # 处理HTTP超时，此时WebSocket可能已经完成
                        end_time = time.time()
                        time_cost = end_time - start_time
                        logger.warning(f"⏱️ HTTP请求超时，耗时: {time_cost:.2f}s")
                        # 等待WebSocket消息处理完成
                        await ws_task
                        # 获取最终回答
                        st.session_state.final_answer = st.session_state.current_answer
                        logger.info(f"✅ WebSocket消息处理完成，最终回答长度: {len(st.session_state.final_answer)} 字符")
                        
                        # 将最终回答添加到聊天历史
                        if st.session_state.final_answer:
                            logger.info(f"📝 将最终回答添加到聊天历史，避免重复显示")
                            st.session_state.messages.append({"role": "assistant", "content": st.session_state.final_answer})
                        # 设置默认的空响应数据
                        res_json = {"code": 200, "data": {}}
                    except Exception as e:
                        # 其他HTTP错误
                        end_time = time.time()
                        time_cost = end_time - start_time
                        logger.error(f"❌ HTTP请求失败，耗时: {time_cost:.2f}s，错误: {str(e)}")
                        # 取消WebSocket任务
                        ws_task.cancel()
                        raise e
                    
                    # 获取数据，无论HTTP请求结果如何
                    data = res_json.get("data", {})
                    if data is None:
                        data = {}
                    intent = data.get("intent", {})
                    sources = data.get("sources", [])
                    all_rows = data.get("all_rows", [])
                    logger.info(f"📊 检索结果统计: 命中 {len(all_rows)} 条数据")
                    
                    # 只有在请求成功或WebSocket完成后，才显示最终结果
                    if res_json.get("code") == 200:
                        status_placeholder.empty()
                        # 提取token统计信息
                        token_stats = data.get("token_stats", {})
                        # 处理嵌套的token_stats结构
                        llm_stats = token_stats.get("llm", {})
                        total_input_tokens = llm_stats.get("total_input_tokens", 0)
                        total_output_tokens = llm_stats.get("total_output_tokens", 0)
                        total_tokens = total_input_tokens + total_output_tokens
                        # 显示结果状态
                        st.success(f"✅ 耗时: {time_cost:.2f}s | 命中数据: {len(all_rows)} 条 | 总Tokens: {total_tokens}")
                        # 显示详细token信息
                        if llm_stats:
                            st.info(f"📊 Token使用情况: 提示词 {total_input_tokens} | 回答 {total_output_tokens} | 总计 {total_tokens}")
                        
                        # ----------------- 检索结果展示 -----------------
                        st.markdown(f"#### 📊 检索到的所有数据 ({len(all_rows)} 条)")
                        if all_rows:
                            # 转换为 DataFrame 展示
                            flat_rows = []
                            for row in all_rows:
                                r_data = row.get("metadata", {}) if isinstance(row, dict) else {}
                                # 过滤掉可能包含LLM回答的字段
                                filtered_data = {}
                                for k, v in r_data.items():
                                    # 跳过可能包含LLM回答的字段
                                    if k not in ['answer', 'llm_answer', 'generated_text', 'response']:
                                        filtered_data[k] = v
                                flat_rows.append(filtered_data)
                                
                            df = pd.DataFrame(flat_rows)
                            logger.info(f"📋 数据转换完成，准备展示 {len(df)} 条记录")
                            
                            # 显示表格
                            st.dataframe(df, width='stretch')
                            
                            # 添加下载按钮，允许下载为Excel文件（使用pandas默认引擎）
                            import io
                            
                            def to_excel(df):
                                output = io.BytesIO()
                                # 使用pandas默认引擎，不需要额外依赖
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    df.to_excel(writer, index=False, sheet_name='检索结果')
                                processed_data = output.getvalue()
                                return processed_data
                            
                            try:
                                # 尝试创建Excel文件
                                excel_data = to_excel(df)
                                logger.info(f"📄 成功生成Excel文件，大小: {len(excel_data)} 字节")
                                
                                # 显示下载按钮
                                st.download_button(
                                    label="📥 下载为XLSX文件",
                                    data=excel_data,
                                    file_name="检索结果.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="download_excel"
                                )
                            except ModuleNotFoundError:
                                # 如果openpyxl也不可用，提供CSV下载作为备选
                                st.warning("Excel导出需要openpyxl模块，当前环境未安装。提供CSV格式下载：")
                                logger.warning("⚠️ openpyxl模块未安装，将使用CSV格式作为备选")
                                
                                # CSV下载功能
                                csv_data = df.to_csv(index=False, encoding='utf-8')
                                logger.info(f"📄 成功生成CSV文件，大小: {len(csv_data)} 字节")
                                st.download_button(
                                    label="📥 下载为CSV文件",
                                    data=csv_data,
                                    file_name="检索结果.csv",
                                    mime="text/csv",
                                    key="download_csv"
                                )
                        else:
                            st.warning("未检索到任何数据")
                            logger.info("⚠️ 未检索到任何数据")
                    else:
                        st.error(f"❌ HTTP请求失败: {response.status_code} - {response.text}")
                        logger.error(f"❌ HTTP请求失败: {response.status_code} - {response.text}")
                except Exception as e:
                    st.error(f"❌ 流式请求失败: {str(e)}")
                    logger.error(f"❌ 流式请求失败: {str(e)}")
                    import traceback
                    logger.error(f"❌ 详细错误信息: {traceback.format_exc()}")
                    st.error(f"❌ 详细错误信息: {traceback.format_exc()}")
                finally:
                    # 关闭WebSocket连接，不显示消息
                    if ws:
                        try:
                            await ws.close()
                            logger.info("🔌 WebSocket连接已关闭")
                        except Exception as e:
                            # 静默处理关闭错误，不向用户显示
                            logger.error(f"❌ 关闭WebSocket连接时出错: {str(e)}")
            
            # 执行流式任务
            asyncio.run(stream_task())
            
            # 流式回答已经通过WebSocket实时显示在assistant消息容器中
            # 并且在stream_task中已经将最终回答添加到了聊天历史
            # 这样确保聊天历史中只有完整的回答，不会有重复显示
        else:
            # 传统非流式模式
            try:
                logger.info(f"📞 开始非流式请求处理，查询: '{prompt}'")
                
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
                logger.info(f"📤 发送非流式HTTP请求到: {api_url}")
                logger.info(f"📋 请求参数: {payload}")
                response = requests.post(api_url, json=payload, headers=headers)
                end_time = time.time()
                time_cost = end_time - start_time
                logger.info(f"📥 收到非流式HTTP响应，状态码: {response.status_code}，耗时: {time_cost:.2f}s")

                if response.status_code == 200:
                    res_json = response.json()
                    logger.info(f"📋 非流式响应数据: {res_json.keys()}")
                    
                    # 提取核心数据
                    if res_json.get("code") == 200:
                        data = res_json.get("data", {})
                        if data is None:
                            data = {}
                        answer = data.get("answer", "未生成回答")
                        intent = data.get("intent", {})
                        sources = data.get("sources", [])
                        all_rows = data.get("all_rows", [])
                        logger.info(f"📊 检索结果统计: 命中 {len(all_rows)} 条数据")
                        
                        status_placeholder.empty()  # 清除"正在加载"
                        # 提取token统计信息
                        token_stats = data.get("token_stats", {})
                        # 处理嵌套的token_stats结构
                        llm_stats = token_stats.get("llm", {})
                        total_input_tokens = llm_stats.get("total_input_tokens", 0)
                        total_output_tokens = llm_stats.get("total_output_tokens", 0)
                        total_tokens = total_input_tokens + total_output_tokens
                        # ----------------- A. 展示 LLM 回答 -----------------
                        st.success(f"✅ 耗时: {time_cost:.2f}s | 命中数据: {len(all_rows)} 条 | 总Tokens: {total_tokens}")
                        # 显示详细token信息
                        if llm_stats:
                            st.info(f"📊 Token使用情况: 提示词 {total_input_tokens} | 回答 {total_output_tokens} | 总计 {total_tokens}")
                        if use_llm:
                            st.markdown("### 🤖 AI 回答")
                            st.markdown(answer)
                            # 将回答加入历史
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            logger.info(f"📝 非流式回答长度: {len(answer)} 字符")
                        else:
                            st.markdown("*LLM 生成已关闭，仅展示检索结果*")
                            logger.info("ℹ️ LLM 生成已关闭，仅展示检索结果")

                        # ----------------- B. 检索结果展示 -----------------
                        st.markdown(f"#### 📊 检索到的所有数据 ({len(all_rows)} 条)")
                        if all_rows:
                            # 转换为 DataFrame 展示
                            flat_rows = []
                            for row in all_rows:
                                r_data = row.get("metadata", {}) if isinstance(row, dict) else {}
                                # 过滤掉可能包含LLM回答的字段
                                filtered_data = {}
                                for k, v in r_data.items():
                                    # 跳过可能包含LLM回答的字段
                                    if k not in ['answer', 'llm_answer', 'generated_text', 'response']:
                                        filtered_data[k] = v
                                flat_rows.append(filtered_data)
                                
                            df = pd.DataFrame(flat_rows)
                            logger.info(f"📋 数据转换完成，准备展示 {len(df)} 条记录")
                            st.dataframe(df, width='stretch')
                            
                            # 添加下载按钮，允许下载为Excel文件（使用pandas默认引擎）
                            import io
                            
                            def to_excel(df):
                                output = io.BytesIO()
                                # 使用pandas默认引擎，不需要额外依赖
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    df.to_excel(writer, index=False, sheet_name='检索结果')
                                processed_data = output.getvalue()
                                return processed_data
                            
                            try:
                                # 尝试创建Excel文件
                                excel_data = to_excel(df)
                                logger.info(f"📄 成功生成Excel文件，大小: {len(excel_data)} 字节")
                                
                                # 显示下载按钮
                                st.download_button(
                                    label="📥 下载为XLSX文件",
                                    data=excel_data,
                                    file_name="检索结果.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key="download_excel"
                                )
                            except ModuleNotFoundError:
                                # 如果openpyxl也不可用，提供CSV下载作为备选
                                st.warning("Excel导出需要openpyxl模块，当前环境未安装。提供CSV格式下载：")
                                logger.warning("⚠️ openpyxl模块未安装，将使用CSV格式作为备选")
                                
                                # CSV下载功能
                                csv_data = df.to_csv(index=False, encoding='utf-8')
                                logger.info(f"📄 成功生成CSV文件，大小: {len(csv_data)} 字节")
                                st.download_button(
                                    label="📥 下载为CSV文件",
                                    data=csv_data,
                                    file_name="检索结果.csv",
                                    mime="text/csv",
                                    key="download_csv"
                                )
                        else:
                            st.warning("未检索到任何数据")
                            logger.info("⚠️ 未检索到任何数据")

                    else:
                        st.error(f"业务错误: {res_json.get('message')}")
                        logger.error(f"❌ 业务错误: {res_json.get('message')}")
                else:
                    st.error(f"HTTP 请求失败: {response.status_code} - {response.text}")
                    logger.error(f"❌ HTTP 请求失败: {response.status_code} - {response.text}")
            except Exception as e:
                st.error(f"发生异常: {str(e)}")
                logger.error(f"❌ 发生异常: {str(e)}")
                import traceback
                logger.error(f"❌ 详细错误信息: {traceback.format_exc()}")
                st.exception(e)