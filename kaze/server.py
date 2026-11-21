"""OpenAI-compatible HTTP server (chat + completions, streaming)."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Annotated, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from kaze.llm import LLM, WORKER_ERROR
from kaze.sampler import MAX_REPETITION_PENALTY, MIN_REPETITION_PENALTY, SamplingParams
from kaze.text import decode_output


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "kaze"
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    temperature: float = Field(default=0.7, ge=0, allow_inf_nan=False)
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    top_k: int = Field(default=0, ge=0)
    repetition_penalty: float = Field(default=1.0, ge=MIN_REPETITION_PENALTY,
                                      le=MAX_REPETITION_PENALTY, allow_inf_nan=False)
    stop: list[str] | str | None = None
    stream: bool = False
    ignore_eos: bool = False
    seed: int | None = Field(default=None, ge=-(2 ** 63), lt=2 ** 64)
    presence_penalty: float = Field(default=0.0, ge=-2, le=2, allow_inf_nan=False)


class CompletionRequest(BaseModel):
    model: str = "kaze"
    # Only single-prompt completions are implemented; never drop batch entries.
    prompt: str | Annotated[list[str], Field(min_length=1, max_length=1)]
    max_tokens: int = Field(default=128, gt=0)
    temperature: float = Field(default=0.7, ge=0, allow_inf_nan=False)
    top_p: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)
    top_k: int = Field(default=0, ge=0)
    repetition_penalty: float = Field(default=1.0, ge=MIN_REPETITION_PENALTY,
                                      le=MAX_REPETITION_PENALTY, allow_inf_nan=False)
    presence_penalty: float = Field(default=0.0, ge=-2, le=2, allow_inf_nan=False)
    seed: int | None = Field(default=None, ge=-(2 ** 63), lt=2 ** 64)
    stop: list[str] | str | None = None
    stream: bool = False
    ignore_eos: bool = False


def _params(max_tokens: int, temperature: float, top_p: float, top_k: int,
            repetition_penalty: float, stop, ignore_eos: bool, seed,
            presence_penalty: float = 0.0) -> SamplingParams:
    stops = stop if isinstance(stop, list) else ([stop] if stop else [])
    return SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                          top_k=top_k or 0, repetition_penalty=repetition_penalty,
                          stop=[s for s in stops if s], ignore_eos=ignore_eos, seed=seed,
                          presence_penalty=presence_penalty)


def create_app(llm: LLM, model_name: str = "kaze") -> FastAPI:
    app = FastAPI(title="Kaze", version="0.1.0")
    t_start = time.time()

    @app.get("/health")
    async def health() -> dict:
        if not llm.is_healthy():
            raise HTTPException(status_code=503, detail=WORKER_ERROR)
        return {"status": "ok", "uptime_s": round(time.time() - t_start, 1)}

    @app.get("/metrics")
    async def metrics() -> dict:
        return llm.metrics()

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [
            {"id": model_name, "object": "model", "created": int(t_start), "owned_by": "kaze"}]}

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest) -> Any:
        msgs = [{"role": m.role, "content": m.content} for m in req.messages]
        params = _params(req.max_tokens or req.max_completion_tokens or 256, req.temperature,
                         req.top_p, req.top_k, req.repetition_penalty, req.stop,
                         req.ignore_eos, req.seed, req.presence_penalty)
        ids = llm._encode(msgs, chat=True)
        return await _serve(llm, ids, params, req.stream, model_name, chat=True)

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest) -> Any:
        params = _params(req.max_tokens, req.temperature, req.top_p, req.top_k,
                         req.repetition_penalty, req.stop, req.ignore_eos, req.seed,
                         req.presence_penalty)
        prompt = req.prompt[0] if isinstance(req.prompt, list) else req.prompt
        ids = llm._encode(prompt, chat=False)
        return await _serve(llm, ids, params, req.stream, model_name, chat=False)

    return app


async def _serve(llm: LLM, ids: list[int], params: SamplingParams, stream: bool,
                 model_name: str, chat: bool) -> Any:
    if not ids:
        raise HTTPException(status_code=422, detail="Prompt must contain at least one token")
    loop = asyncio.get_running_loop()
    try:
        # a non-streaming request only needs the final output: skip the per-token wakeups
        if stream or not hasattr(llm, "stream_final_only"):
            q = llm.stream(ids, params)
        else:
            q = llm.stream_final_only(ids, params)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from None
    except RuntimeError:
        raise HTTPException(status_code=503, detail=WORKER_ERROR) from None
    cid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
    created = int(time.time())
    if not stream:
        # the queue path beats a completion event at high concurrency (measured
        # 6,789 vs 5,638 tok/s at 32 streams): waking 32 threads at once makes them
        # fight over the GIL for tokenizer.decode + JSON.  The library API keeps the
        # event path (LLM.generate) because it wins at low concurrency.
        t_enter = time.perf_counter()
        tokens: list[int] = []
        t_first = None
        while True:
            out = await loop.run_in_executor(None, q.get)
            if t_first is None:
                t_first = time.perf_counter()
            if out.error:
                raise HTTPException(status_code=503, detail=WORKER_ERROR)
            if out.tokens:
                tokens.extend(out.tokens)
            if out.finished:
                break
        t_done = time.perf_counter()
        text = decode_output(llm.tokenizer, tokens, params.stop)
        t_text = time.perf_counter()
        prof = getattr(llm, "_prof", None)
        if prof is not None:
            # wait = submission -> first token, run = first -> last token,
            # tail = last token -> response text ready
            prof["req_n"] = prof.get("req_n", 0) + 1
            prof["req_wait_s"] = prof.get("req_wait_s", 0.0) + (t_first - t_enter)
            prof["req_run_s"] = prof.get("req_run_s", 0.0) + (t_done - t_first)
            prof["req_tail_s"] = prof.get("req_tail_s", 0.0) + (t_text - t_done)
        reason = out.finish_reason or "stop"
        if chat:
            return {"id": cid, "object": "chat.completion", "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                 "finish_reason": reason}],
                    "usage": {"prompt_tokens": len(ids), "completion_tokens": len(tokens),
                              "total_tokens": len(ids) + len(tokens)}}
        return {"id": cid, "object": "text_completion", "created": created, "model": model_name,
                "choices": [{"index": 0, "text": text, "finish_reason": reason}],
                "usage": {"prompt_tokens": len(ids), "completion_tokens": len(tokens),
                          "total_tokens": len(ids) + len(tokens)}}

    async def gen():
        tokens: list[int] = []
        sent = 0
        while True:
            out = await loop.run_in_executor(None, q.get)
            if out.error:
                payload = {"error": {"message": WORKER_ERROR, "type": "server_error"}}
                yield f"event: error\ndata: {json.dumps(payload)}\n\n"
                yield "data: [DONE]\n\n"
                break
            if out.tokens:
                tokens.extend(out.tokens)
            text = decode_output(llm.tokenizer, tokens, params.stop, finished=out.finished)
            delta = text[sent:]
            sent = len(text)
            if delta:
                if chat:
                    payload = {"id": cid, "object": "chat.completion.chunk",
                               "created": created, "model": model_name,
                               "choices": [{"index": 0, "delta": {"content": delta},
                                            "finish_reason": None}]}
                else:
                    payload = {"id": cid, "object": "text_completion", "created": created,
                               "model": model_name,
                               "choices": [{"index": 0, "text": delta, "finish_reason": None}]}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
            if out.finished:
                if chat:
                    payload = {"id": cid, "object": "chat.completion.chunk", "created": created,
                               "model": model_name,
                               "choices": [{"index": 0, "delta": {},
                                            "finish_reason": out.finish_reason or "stop"}]}
                else:
                    payload = {"id": cid, "object": "text_completion", "created": created,
                               "model": model_name,
                               "choices": [{"index": 0, "text": "",
                                            "finish_reason": out.finish_reason or "stop"}]}
                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                break

    return StreamingResponse(gen(), media_type="text/event-stream")
