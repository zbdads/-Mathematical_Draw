"""Math-model generation API."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from src.evaluation.hhc_modeling_eval import run_hhc_modeling_eval
from src.modeling.model_builder import ModelingCancelledError, generate_harness_draft, generate_math_model
from src.modeling.platemo_codegen import generate_platemo_code as build_platemo_code
from web.services.job_store import job_store
from web.services.query_service import get_service
from web.utils import image_path_to_url, serialize_result

router = APIRouter()
logger = logging.getLogger(__name__)


class GenerateModelRequest(BaseModel):
    problem: str
    top_k: int | None = None
    paper_id: str | None = None
    use_blueprint: bool = False
    render_formulas: bool = False
    academic_mode: bool = False
    agent_mode: bool = False
    agent_max_rounds: int | None = None
    agent_quality_threshold: float | None = None
    max_tokens: int | None = None
    generate_platemo_code: bool = False
    platemo_root: str | None = None
    platemo_class_name: str | None = None
    write_platemo_file: bool = False


class GeneratePlatemoCodeRequest(BaseModel):
    problem: str = ""
    model: dict[str, Any] | None = None
    harness_draft: dict[str, Any] | None = None
    problem_spec: dict[str, Any] | None = None
    platemo_root: str | None = None
    class_name: str | None = None
    write_file: bool = True


class HhcEvalRequest(BaseModel):
    top_k: int = 4
    paper_id: str | None = None
    full_llm: bool = False
    max_tokens: int | None = None
    limit_cases: int | None = None
    case_ids: list[str] = []
    fast_llm_eval: bool = True
    save_report: bool = True


def _enrich_figures(figure_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for result in figure_results:
        url = image_path_to_url(result.get("image_path"))
        if url is not None:
            result = {**result, "image_url": url}
        out.append(result)
    return out


def _serialize_modeling_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem": result["problem"],
        "retrieval_query": result["retrieval_query"],
        "skill": result.get("skill", ""),
        "skill_description": result.get("skill_description", ""),
        "generation_mode": result.get("generation_mode", ""),
        "modeling_plan": result.get("modeling_plan"),
        "plan_output": result.get("plan_output", ""),
        "plan_error": result.get("plan_error", ""),
        "problem_spec": result.get("problem_spec"),
        "harness_draft": result.get("harness_draft"),
        "modeling_blueprint": result.get("modeling_blueprint"),
        "blueprint_output": result.get("blueprint_output", ""),
        "blueprint_error": result.get("blueprint_error", ""),
        "paper_id": result["paper_id"],
        "top_k": result["top_k"],
        "text_results": [serialize_result(r) for r in result["text_results"]],
        "figure_results": _enrich_figures(
            [serialize_result(r) for r in result["figure_results"]]
        ),
        "model": result["model"],
        "raw_output": result["raw_output"],
        "repair_output": result["repair_output"],
        "revision_output": result.get("revision_output", ""),
        "revision_note": result.get("revision_note", ""),
        "parse_error": result["parse_error"],
        "warnings": result["warnings"],
        "model_verification": result.get("model_verification"),
        "model_quality": result.get("model_quality"),
        "agent_mode": result.get("agent_mode", False),
        "agent_trace": result.get("agent_trace", []),
        "agent_terminate_reason": result.get("agent_terminate_reason", ""),
        "code_generation": result.get("code_generation"),
    }


def _job_payload(job_id: str) -> dict[str, Any]:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job 不存在或已被清理")
    return {
        "job_id": job_id,
        "status": job.get("status", "unknown"),
        "kind": job.get("kind", ""),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at"),
        "terminal_at": job.get("terminal_at"),
        "stage": job.get("stage", ""),
        "message": job.get("message", ""),
        "progress": job.get("progress", 0.0),
        "cancel_requested": bool(job.get("cancel_requested")),
        "error": job.get("error"),
        "result": job.get("result"),
    }


def _request_dump(req: BaseModel) -> dict[str, Any]:
    if hasattr(req, "model_dump"):
        return req.model_dump()
    return req.dict()


class JobCancelledError(RuntimeError):
    pass


def _set_job_progress(
    job_id: str,
    *,
    stage: str,
    message: str,
    progress: float | None = None,
) -> None:
    job_store.update_progress(job_id, stage=stage, message=message, progress=progress)


def _raise_if_cancelled(job_id: str) -> None:
    if job_store.is_cancel_requested(job_id):
        raise JobCancelledError("job cancelled by user")


def _mark_job_cancelled(job_id: str) -> None:
    job_store.update_status(job_id, "cancelled")
    job_store.push_event(job_id, {"type": "cancelled", "message": "任务已取消"})


async def _run_modeling_generate_job(job_id: str, req: GenerateModelRequest) -> None:
    job_store.update_status(job_id, "running")
    job = job_store.get(job_id)
    if job is not None:
        job["kind"] = "modeling_generate"
    try:
        _set_job_progress(
            job_id,
            stage="preparing",
            message="正在准备检索服务",
            progress=0.08,
        )
        _raise_if_cancelled(job_id)
        service, _ = await get_service()
        _raise_if_cancelled(job_id)
        generation_kwargs: dict[str, Any] = {}
        if req.academic_mode:
            generation_kwargs.update(
                {
                    "max_tokens": req.max_tokens or 2200,
                    "reasoning_effort": "xhigh",
                    "timeout": 420.0,
                    "client_max_retries": 1,
                    "generation_attempts": 2,
                    "use_blueprint": True,
                }
            )
        else:
            generation_kwargs.update(
                {
                    "max_tokens": req.max_tokens,
                    "use_blueprint": req.use_blueprint,
                }
            )
        if req.agent_mode:
            generation_kwargs.update(
                {
                    "agent_mode": True,
                    "agent_max_rounds": 0 if req.agent_max_rounds is None else req.agent_max_rounds,
                    "agent_quality_threshold": 0.86 if req.agent_quality_threshold is None else req.agent_quality_threshold,
                    "skip_revision": True,
                }
            )
        if req.generate_platemo_code:
            generation_kwargs.update(
                {
                    "generate_platemo_code": True,
                    "platemo_root": req.platemo_root,
                    "platemo_class_name": req.platemo_class_name,
                    "write_platemo_file": req.write_platemo_file,
                }
            )

        def push_progress(stage: str, message: str, progress: float) -> None:
            _set_job_progress(job_id, stage=stage, message=message, progress=progress)

        result: dict[str, Any] = await asyncio.to_thread(
            generate_math_model,
            req.problem,
            top_k=req.top_k,
            paper_id=req.paper_id,
            service=service,
            progress_callback=push_progress,
            cancel_check=lambda: job_store.is_cancel_requested(job_id),
            **generation_kwargs,
        )
        _raise_if_cancelled(job_id)
        _set_job_progress(
            job_id,
            stage="serializing",
            message="正在整理模型、证据和校验结果",
            progress=0.92,
        )
        job = job_store.get(job_id)
        if job is not None:
            job["result"] = _serialize_modeling_result(result)
            job["stage"] = "done"
            job["message"] = "完整生成已完成"
        job_store.update_status(job_id, "done")
        job_store.push_event(job_id, {"type": "done"})
    except (JobCancelledError, ModelingCancelledError):
        _mark_job_cancelled(job_id)
    except Exception as exc:
        logger.exception("[modeling] generate job=%s failed", job_id)
        job = job_store.get(job_id)
        if job is not None:
            job["error"] = str(exc) or exc.__class__.__name__
            job["stage"] = "error"
            job["message"] = job["error"]
        job_store.update_status(job_id, "error")
        job_store.push_event(job_id, {"type": "error", "message": str(exc)})


async def _run_hhc_eval_job(job_id: str, req: HhcEvalRequest) -> None:
    job_store.update_status(job_id, "running")
    job = job_store.get(job_id)
    if job is not None:
        job["kind"] = "hhc_eval"
    try:
        _set_job_progress(
            job_id,
            stage="preparing",
            message="正在准备 HHC 评测任务",
            progress=0.08,
        )
        _raise_if_cancelled(job_id)
        service, _ = await get_service()
        mode = "完整大模型评测" if req.full_llm else "Harness 公式评测"
        _set_job_progress(
            job_id,
            stage="evaluating",
            message=f"正在运行 {mode}",
            progress=0.25,
        )
        _raise_if_cancelled(job_id)
        result: dict[str, Any] = await asyncio.to_thread(
            run_hhc_modeling_eval,
            top_k=req.top_k,
            paper_id=req.paper_id,
            service=service,
            full_llm=req.full_llm,
            max_tokens=req.max_tokens,
            limit_cases=req.limit_cases,
            case_ids=req.case_ids,
            fast_llm_eval=req.fast_llm_eval,
            save_report=req.save_report,
        )
        _raise_if_cancelled(job_id)
        _set_job_progress(
            job_id,
            stage="serializing",
            message="正在整理 HHC 评测报告",
            progress=0.92,
        )
        job = job_store.get(job_id)
        if job is not None:
            job["result"] = result
            job["stage"] = "done"
            job["message"] = "HHC 评测已完成"
        job_store.update_status(job_id, "done")
        job_store.push_event(job_id, {"type": "done"})
    except JobCancelledError:
        _mark_job_cancelled(job_id)
    except Exception as exc:
        logger.exception("[modeling] hhc eval job=%s failed", job_id)
        job = job_store.get(job_id)
        if job is not None:
            job["error"] = str(exc) or exc.__class__.__name__
            job["stage"] = "error"
            job["message"] = job["error"]
        job_store.update_status(job_id, "error")
        job_store.push_event(job_id, {"type": "error", "message": str(exc)})


@router.post("/modeling/generate")
async def modeling_generate(req: GenerateModelRequest) -> dict[str, Any]:
    service, _ = await get_service()
    try:
        result: dict[str, Any] = await asyncio.to_thread(
            generate_math_model,
            req.problem,
            top_k=req.top_k,
            paper_id=req.paper_id,
            service=service,
            use_blueprint=req.academic_mode or req.use_blueprint,
            max_tokens=req.max_tokens or (2200 if req.academic_mode else None),
            reasoning_effort="xhigh" if req.academic_mode else None,
            timeout=420.0 if req.academic_mode else None,
            client_max_retries=1 if req.academic_mode else None,
            generation_attempts=2 if req.academic_mode else 3,
            agent_mode=req.agent_mode,
            agent_max_rounds=0 if req.agent_max_rounds is None else req.agent_max_rounds,
            agent_quality_threshold=0.86 if req.agent_quality_threshold is None else req.agent_quality_threshold,
            skip_revision=req.agent_mode,
            generate_platemo_code=req.generate_platemo_code,
            platemo_root=req.platemo_root,
            platemo_class_name=req.platemo_class_name,
            write_platemo_file=req.write_platemo_file,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc) or exc.__class__.__name__) from exc
    return _serialize_modeling_result(result)


@router.post("/modeling/platemo-code")
async def modeling_platemo_code(req: GeneratePlatemoCodeRequest) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(
            build_platemo_code,
            problem=req.problem,
            model=req.model,
            harness_draft=req.harness_draft,
            problem_spec=req.problem_spec,
            platemo_root=req.platemo_root,
            class_name=req.class_name,
            write_file=req.write_file,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc) or exc.__class__.__name__) from exc


@router.post("/modeling/generate/jobs", status_code=202)
async def modeling_generate_job(
    req: GenerateModelRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    if not req.problem.strip():
        raise HTTPException(status_code=422, detail="problem 不能为空")
    job_id = job_store.create()
    job = job_store.get(job_id)
    if job is not None:
        job["kind"] = "modeling_generate"
        job["request"] = _request_dump(req)
    background_tasks.add_task(_run_modeling_generate_job, job_id, req)
    return _job_payload(job_id)


@router.get("/modeling/generate/jobs/{job_id}")
async def modeling_generate_job_status(job_id: str) -> dict[str, Any]:
    return _job_payload(job_id)


@router.post("/modeling/generate/jobs/{job_id}/cancel")
async def modeling_generate_job_cancel(job_id: str) -> dict[str, Any]:
    if job_store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job 不存在或已被清理")
    job_store.request_cancel(job_id)
    return _job_payload(job_id)


@router.post("/modeling/harness")
async def modeling_harness(req: GenerateModelRequest) -> dict[str, Any]:
    service, _ = await get_service()
    try:
        result: dict[str, Any] = await asyncio.to_thread(
            generate_harness_draft,
            req.problem,
            top_k=req.top_k,
            paper_id=req.paper_id,
            service=service,
            render_formulas=req.render_formulas,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc) or exc.__class__.__name__) from exc
    return {
        "problem": result["problem"],
        "retrieval_query": result["retrieval_query"],
        "skill": result.get("skill", ""),
        "skill_description": result.get("skill_description", ""),
        "generation_mode": result.get("generation_mode", ""),
        "paper_id": result["paper_id"],
        "top_k": result["top_k"],
        "text_results": [serialize_result(r) for r in result["text_results"]],
        "figure_results": _enrich_figures(
            [serialize_result(r) for r in result["figure_results"]]
        ),
        "harness_draft": result.get("harness_draft"),
        "model": result.get("model"),
        "model_verification": result.get("model_verification"),
        "model_quality": result.get("model_quality"),
        "warnings": result.get("warnings", []),
    }


@router.post("/modeling/eval/hhc")
async def modeling_eval_hhc(req: HhcEvalRequest) -> dict[str, Any]:
    service, _ = await get_service()
    try:
        result: dict[str, Any] = await asyncio.to_thread(
            run_hhc_modeling_eval,
            top_k=req.top_k,
            paper_id=req.paper_id,
            service=service,
            full_llm=req.full_llm,
            max_tokens=req.max_tokens,
            limit_cases=req.limit_cases,
            case_ids=req.case_ids,
            fast_llm_eval=req.fast_llm_eval,
            save_report=req.save_report,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc) or exc.__class__.__name__) from exc
    return result


@router.post("/modeling/eval/hhc/jobs", status_code=202)
async def modeling_eval_hhc_job(
    req: HhcEvalRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    job_id = job_store.create()
    job = job_store.get(job_id)
    if job is not None:
        job["kind"] = "hhc_eval"
        job["request"] = _request_dump(req)
    background_tasks.add_task(_run_hhc_eval_job, job_id, req)
    return _job_payload(job_id)


@router.get("/modeling/eval/hhc/jobs/{job_id}")
async def modeling_eval_hhc_job_status(job_id: str) -> dict[str, Any]:
    return _job_payload(job_id)


@router.post("/modeling/eval/hhc/jobs/{job_id}/cancel")
async def modeling_eval_hhc_job_cancel(job_id: str) -> dict[str, Any]:
    if job_store.get(job_id) is None:
        raise HTTPException(status_code=404, detail="job 不存在或已被清理")
    job_store.request_cancel(job_id)
    return _job_payload(job_id)
