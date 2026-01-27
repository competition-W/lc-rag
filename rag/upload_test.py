import requests

# 文件路径
file_path = r'D:\LC-BIO\lc-rag-V2\rag\test_data\20260114-单细胞时空组学标准化材料表格梳理.xlsx'

# API端点
url = 'http://127.0.0.1:8090/documents/upload'

# 请求头
headers = {
    'accept': 'application/json',
    'X-User-ID': 'test_user',
    'X-Department': 'market'
}

# 表单数据
files = {
    'file': open(file_path, 'rb')
}

form_data = {
    'target_dept': 'market'
}

try:
    # 发送请求
    response = requests.post(url, headers=headers, files=files, data=form_data)
    
    # 打印响应结果
    print(f"状态码: {response.status_code}")
    print(f"响应内容: {response.text}")
    
    # 解析JSON响应
    if response.status_code == 200:
        data = response.json()
        print(f"任务ID: {data.get('data', {}).get('task_id')}")
        print(f"文件ID: {data.get('data', {}).get('file_id')}")
        print(f"状态: {data.get('data', {}).get('status')}")
        print(f"消息: {data.get('message')}")
except Exception as e:
    print(f"上传失败: {e}")
finally:
    # 关闭文件
    files['file'].close()
    