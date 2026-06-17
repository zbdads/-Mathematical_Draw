"""In-memory ingest job 状态表。"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from typing import Any

_TERMINAL_TTL = 1800  # 30 分钟


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}

    def create(self) -> str:
        job_id = str(uuid.uuid4())
        now = time.time()
        self._jobs[job_id] = {
            "status": "pending",
            "queue": asyncio.Queue(maxsize=200),
            "events_log": deque(maxlen=200),
            "created_at": now,
            "updated_at": now,
            "terminal_at": None,
            "error": None,
            "result": None,
            "stage": "pending",
            "message": "任务已创建，等待开始",
            "progress": 0.0,
            "cancel_requested": False,
        }
        return job_id

    def get(self, job_id: str) -> dict[str, Any] | None:
        return self._jobs.get(job_id)

    def update_status(self, job_id: str, status: str) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job["status"] = status
        job["updated_at"] = time.time()
        if status in ("done", "error", "cancelled"):
            job["terminal_at"] = time.time()
            if status == "done":
                job["progress"] = 1.0
            elif status == "cancelled":
                job["message"] = "任务已取消"

    def update_progress(
        self,
        job_id: str,
        *,
        stage: str,
        message: str,
        progress: float | None = None,
    ) -> None:
        job = self._jobs.get(job_id)
        if job is None:
            return
        job["stage"] = stage
        job["message"] = message
        if progress is not None:
            job["progress"] = max(0.0, min(1.0, float(progress)))
        job["updated_at"] = time.time()
        self.push_event(
            job_id,
            {
                "type": "progress",
                "stage": job["stage"],
                "message": job["message"],
                "progress": job["progress"],
            },
        )

    def request_cancel(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False
        if job["status"] in ("done", "error", "cancelled"):
            return False
        job["cancel_requested"] = True
        job["updated_at"] = time.time()
        if job["status"] in ("pending", "running"):
            job["status"] = "cancelling"
            job["message"] = "取消请求已发送，等待当前阶段停止"
        self.push_event(
            job_id,
            {
                "type": "cancelling",
                "message": job["message"],
                "stage": job.get("stage", ""),
                "progress": job.get("progress", 0.0),
            },
        )
        return True

    def is_cancel_requested(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        return bool(job and job.get("cancel_requested"))

    def push_event(self, job_id: str, event: dict) -> None:
        """线程安全地向队列推送事件（从 worker 线程调用时需配合 call_soon_threadsafe）。"""
        job = self._jobs.get(job_id)
        if job is None:
            return
        job["events_log"].append(event)
        q: asyncio.Queue = job["queue"]
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                q.put_nowait(event)
            except asyncio.QueueFull:
                pass

    def purge_stale(self) -> None:
        now = time.time()
        stale = [
            jid
            for jid, job in self._jobs.items()
            if job["terminal_at"] is not None and now - job["terminal_at"] > _TERMINAL_TTL
        ]
        for jid in stale:
            del self._jobs[jid]


job_store = JobStore()
