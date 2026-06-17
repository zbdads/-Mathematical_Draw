"""GET /api/status — 返回模型加载状态。"""

from __future__ import annotations

from fastapi import APIRouter

from web.services.query_service import get_load_status

router = APIRouter()


@router.get("/status")
def status() -> dict:
    """返回 { state: idle|loading|ready|error, message: str }。"""
    return get_load_status()
