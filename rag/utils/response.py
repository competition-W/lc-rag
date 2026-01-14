#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一响应格式
"""
from typing import Any, Optional, Dict
from pydantic import BaseModel


class APIResponse(BaseModel):
    """统一API响应格式"""
    code: int = 200
    message: str = "success"
    data: Optional[Any] = None


def success(data: Any = None, message: str = "操作成功") -> Dict:
    """成功响应"""
    return APIResponse(code=200, message=message, data=data).model_dump()


def error(message: str = "操作失败", code: int = 400, data: Any = None) -> Dict:
    """错误响应"""
    return APIResponse(code=code, message=message, data=data).model_dump()


def not_found(message: str = "资源不存在") -> Dict:
    """404响应"""
    return APIResponse(code=404, message=message).model_dump()


def unauthorized(message: str = "未授权访问") -> Dict:
    """401响应"""
    return APIResponse(code=401, message=message).model_dump()


def server_error(message: str = "服务器内部错误") -> Dict:
    """500响应"""
    return APIResponse(code=500, message=message).model_dump()