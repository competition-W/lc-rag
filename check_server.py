#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查服务器状态脚本
"""

import requests


def check_server():
    """检查服务器是否运行"""
    try:
        response = requests.get('http://localhost:8000')
        print(f"Server is running, status code: {response.status_code}")
        return True
    except requests.ConnectionError:
        print("Server is not running")
        return False


if __name__ == "__main__":
    check_server()
