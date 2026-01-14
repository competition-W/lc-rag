#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
权限验证工具
"""
from typing import Optional, List
from fastapi import HTTPException, Header, status
import logging

from config import settings, get_collection_name, is_admin_user, DEPARTMENT_COLLECTIONS

logger = logging.getLogger(__name__)


class PermissionDenied(HTTPException):
    """权限拒绝异常"""
    def __init__(self, detail: str = "Permission Denied"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail
        )


class AuthContext:
    """请求认证上下文"""
    
    def __init__(
        self,
        user_id: str,
        department: str,
        user_role: Optional[str] = None
    ):
        self.user_id = user_id
        self.department = department
        self.user_role = user_role or "user"
        self.is_admin = is_admin_user(self.user_role)
    
    def can_access_department(self, target_department: str) -> bool:
        """检查是否可访问目标部门"""
        # 管理员可访问所有部门
        if self.is_admin and settings.ENABLE_CROSS_DEPARTMENT:
            return True
        
        # 普通用户只能访问自己的部门
        return self.department == target_department
    
    def get_accessible_collections(self) -> List[str]:
        """获取可访问的Collection列表"""
        if self.is_admin and settings.ENABLE_CROSS_DEPARTMENT:
            # 管理员：所有Collection
            return list(DEPARTMENT_COLLECTIONS.values())
        else:
            # 普通用户：仅自己部门
            return [get_collection_name(self.department)]


async def get_auth_context(
    user_id: Optional[str] = Header(None, alias=settings.HEADER_USER_ID),
    department: Optional[str] = Header(None, alias=settings.HEADER_DEPARTMENT),
    user_role: Optional[str] = Header(None, alias=settings.HEADER_USER_ROLE)
) -> AuthContext:
    """
    从请求头提取认证信息
    
    这是一个依赖注入函数，在FastAPI路由中使用：
    @router.post("/upload")
    async def upload(auth: AuthContext = Depends(get_auth_context)):
        ...
    """
    # 禁用权限模式（开发阶段）
    if settings.PERMISSION_MODE == "disabled":
        logger.warning("⚠️  权限验证已禁用（开发模式）")
        return AuthContext(
            user_id=user_id or "dev_user",
            department=department or "market",
            user_role="admin"  # 开发模式默认管理员
        )
    
    # 验证必需字段
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing header: {settings.HEADER_USER_ID}"
        )
    
    if not department:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing header: {settings.HEADER_DEPARTMENT}"
        )
    
    # 验证部门是否存在
    if department not in DEPARTMENT_COLLECTIONS:
        raise PermissionDenied(f"Invalid department: {department}")
    
    # 严格模式：验证Token（TODO）
    if settings.PERMISSION_MODE == "strict":
        # await verify_jwt_token(request.headers.get("Authorization"))
        pass
    
    logger.info(f"✅ 认证通过: user={user_id}, dept={department}, role={user_role}")
    
    return AuthContext(
        user_id=user_id,
        department=department,
        user_role=user_role
    )


def require_department_access(auth: AuthContext, target_department: str):
    """
    验证部门访问权限（装饰器辅助函数）
    
    使用示例：
    require_department_access(auth, "market")
    """
    if not auth.can_access_department(target_department):
        raise PermissionDenied(
            f"User {auth.user_id} cannot access department: {target_department}"
        )


async def verify_upload_permission(
    auth: AuthContext,
    target_department: str
) -> bool:
    """
    验证上传权限
    
    Args:
        auth: 认证上下文
        target_department: 目标部门
        
    Returns:
        是否有权限
    """
    # 检查部门访问权限
    if not auth.can_access_department(target_department):
        return False
    
    # TODO: 添加更细粒度的权限检查
    # - 检查用户是否有上传权限（role-based）
    # - 检查文件类型权限
    # - 检查配额限制
    
    return True


async def verify_query_permission(
    auth: AuthContext,
    target_departments: List[str]
) -> List[str]:
    """
    验证查询权限，返回实际可查询的部门列表
    
    Args:
        auth: 认证上下文
        target_departments: 请求查询的部门列表
        
    Returns:
        过滤后的可查询部门列表
    """
    accessible = []
    
    for dept in target_departments:
        if auth.can_access_department(dept):
            accessible.append(dept)
        else:
            logger.warning(
                f"User {auth.user_id} attempted to access {dept} - DENIED"
            )
    
    if not accessible:
        raise PermissionDenied("No accessible departments in the request")
    
    return accessible