from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from llms.proxy.auth import require_admin
from llms.proxy.config import Settings, settings_from_app
from llms.proxy.store import ApiKey, ModelEntry, Store, is_secret_key

router = APIRouter()


def _store(settings: Settings) -> Store:
    return Store(data_dir=Path(settings.data_dir))


def _usage(request: Request):
    return request.app.state.usage.snapshot()


class KeyBody(BaseModel):
    key: str = ""
    label: str = ""
    enabled: bool = True


class KeyPatch(BaseModel):
    label: str | None = None
    enabled: bool | None = None


class ModelBody(BaseModel):
    id: str = ""
    label: str = ""
    enabled: bool = True


class ModelPatch(BaseModel):
    label: str | None = None
    enabled: bool | None = None


@router.get("/api/admin/keys")
async def list_keys(
    request: Request,
    settings: Settings = Depends(settings_from_app),
    _admin: str = Depends(require_admin),
):
    keys = _store(settings).load_keys()
    usage_keys = _usage(request).get("keys", {})
    return {
        "keys": [
            {
                "key": k.key,
                "label": k.label,
                "enabled": k.enabled,
                "usage": usage_keys.get(
                    k.key,
                    {
                        "requests": 0,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "cached_tokens": 0,
                        "reasoning_tokens": 0,
                        "models": {},
                    },
                ),
            }
            for k in keys
        ]
    }


@router.post("/api/admin/keys", status_code=201)
async def create_key(
    body: KeyBody,
    settings: Settings = Depends(settings_from_app),
    _admin: str = Depends(require_admin),
):
    if not body.key:
        raise HTTPException(status_code=400, detail="key is required")
    if not is_secret_key(body.key):
        raise HTTPException(
            status_code=400,
            detail="api keys must start with sk- (ak- is affinity, not auth)",
        )
    store = _store(settings)
    keys = store.load_keys()
    if any(k.key == body.key for k in keys):
        raise HTTPException(status_code=409, detail="key already exists")
    keys.append(ApiKey(key=body.key, label=body.label, enabled=body.enabled))
    store.save_keys(keys)
    return {"key": body.key, "label": body.label, "enabled": body.enabled}


@router.put("/api/admin/keys/{key:path}")
async def update_key(
    key: str,
    body: KeyPatch,
    settings: Settings = Depends(settings_from_app),
    _admin: str = Depends(require_admin),
):
    store = _store(settings)
    keys = store.load_keys()
    for k in keys:
        if k.key == key:
            if body.label is not None:
                k.label = body.label
            if body.enabled is not None:
                k.enabled = body.enabled
            store.save_keys(keys)
            return {"key": k.key, "label": k.label, "enabled": k.enabled}
    raise HTTPException(status_code=404, detail="key not found")


@router.delete("/api/admin/keys/{key:path}")
async def delete_key(
    key: str,
    settings: Settings = Depends(settings_from_app),
    _admin: str = Depends(require_admin),
):
    store = _store(settings)
    keys = [k for k in store.load_keys() if k.key != key]
    if len(keys) == len(store.load_keys()):
        raise HTTPException(status_code=404, detail="key not found")
    store.save_keys(keys)
    return {"status": "ok"}


@router.get("/api/admin/models")
async def list_models(
    settings: Settings = Depends(settings_from_app),
    _admin: str = Depends(require_admin),
):
    return {
        "models": [
            {"id": m.id, "label": m.label, "enabled": m.enabled}
            for m in _store(settings).load_models()
        ]
    }


@router.post("/api/admin/models", status_code=201)
async def create_model(
    body: ModelBody,
    settings: Settings = Depends(settings_from_app),
    _admin: str = Depends(require_admin),
):
    if not body.id:
        raise HTTPException(status_code=400, detail="id is required")
    store = _store(settings)
    models = store.load_models()
    if any(m.id == body.id for m in models):
        raise HTTPException(status_code=409, detail="model already exists")
    models.append(ModelEntry(id=body.id, label=body.label, enabled=body.enabled))
    store.save_models(models)
    return {"id": body.id, "label": body.label, "enabled": body.enabled}


@router.put("/api/admin/models/{model_id:path}")
async def update_model(
    model_id: str,
    body: ModelPatch,
    settings: Settings = Depends(settings_from_app),
    _admin: str = Depends(require_admin),
):
    store = _store(settings)
    models = store.load_models()
    for m in models:
        if m.id == model_id:
            if body.label is not None:
                m.label = body.label
            if body.enabled is not None:
                m.enabled = body.enabled
            store.save_models(models)
            return {"id": m.id, "label": m.label, "enabled": m.enabled}
    raise HTTPException(status_code=404, detail="model not found")


@router.delete("/api/admin/models/{model_id:path}")
async def delete_model(
    model_id: str,
    settings: Settings = Depends(settings_from_app),
    _admin: str = Depends(require_admin),
):
    store = _store(settings)
    models = [m for m in store.load_models() if m.id != model_id]
    if len(models) == len(store.load_models()):
        raise HTTPException(status_code=404, detail="model not found")
    store.save_models(models)
    return {"status": "ok"}


@router.get("/api/admin/usage")
async def get_usage(request: Request, _admin: str = Depends(require_admin)):
    return _usage(request)
