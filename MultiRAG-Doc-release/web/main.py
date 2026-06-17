"""FastAPI app 入口：挂载路由、静态文件、图片服务。"""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.config import CFG
from web.routers import agent, ingest, modeling, papers, query, status
from web.services.job_store import job_store
from web.services.query_service import preload_service

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="MultiRAG-Doc Web API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 图片静态服务
_fig_dir = CFG.paths.index_dir / "figures"
_fig_dir.mkdir(parents=True, exist_ok=True)
app.mount("/figures", StaticFiles(directory=str(_fig_dir)), name="figures")

# API 路由
app.include_router(status.router, prefix="/api")
app.include_router(papers.router, prefix="/api")
app.include_router(query.router, prefix="/api")
app.include_router(agent.router, prefix="/api")
app.include_router(modeling.router, prefix="/api")
app.include_router(ingest.router, prefix="/api")

# 前端静态文件（index.html 兜底放在最后，避免遮盖 API）
_static_dir = CFG.paths.root / "web" / "static"
app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="static")


@app.on_event("startup")
async def _startup() -> None:
    """启动时：预加载模型 + 启动 ingest job 定期清理。"""
    asyncio.create_task(preload_service())

    async def _cleanup_loop() -> None:
        while True:
            await asyncio.sleep(300)
            job_store.purge_stale()

    asyncio.create_task(_cleanup_loop())
