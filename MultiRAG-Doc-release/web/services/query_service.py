"""QueryService 单例管理（懒加载 + ingest 后失效）。"""

from __future__ import annotations

import asyncio
import logging

from src.pipeline.query import QueryService

logger = logging.getLogger(__name__)

_service: QueryService | None = None
_lock = asyncio.Lock()
_generation: int = 0

# 模型加载状态：idle / loading / ready / error
_load_status: dict = {"state": "idle", "message": ""}


def get_load_status() -> dict:
    """返回当前模型加载状态快照。"""
    return dict(_load_status)


def get_embedder_if_loaded():
    """若 QueryService 已就绪，返回其 TextEmbedder 供 ingest 复用；否则返回 None。"""
    return _service.embedder if _service is not None else None


async def get_service() -> tuple[QueryService, int]:
    """懒加载 QueryService 单例，返回 (service, generation) 快照。"""
    global _service, _generation
    async with _lock:
        if _service is None:
            gen = _generation
            logger.info("[query_service] 加载 QueryService（generation=%d）…", gen)
            _service = await asyncio.to_thread(QueryService.from_disk)
            logger.info("[query_service] QueryService 加载完成")
        return _service, _generation


async def preload_service() -> None:
    """后台预加载 QueryService，供 startup 事件调用。"""
    global _load_status
    _load_status = {"state": "loading", "message": ""}
    try:
        await get_service()
        _load_status = {"state": "ready", "message": ""}
        logger.info("[query_service] 预加载完成")
    except Exception as e:
        _load_status = {"state": "error", "message": str(e)}
        logger.error("[query_service] 预加载失败: %s", e)


async def invalidate_service() -> None:
    """ingest 成功后调用，使下一次 get_service() 重新从磁盘加载。"""
    global _service, _generation
    old: QueryService | None = None
    async with _lock:
        _generation += 1
        old = _service
        _service = None
        logger.info("[query_service] QueryService 已失效（new generation=%d）", _generation)
    # 在 lock 外关闭旧实例，不阻塞新请求
    if old is not None:
        try:
            old.close()
        except Exception:
            pass
