export interface ModelUsage {
  requests: number;
  input_tokens: number;
  output_tokens: number;
  cached_tokens: number;
  reasoning_tokens: number;
}

export interface KeyUsage extends ModelUsage {
  models: Record<string, ModelUsage>;
}

export interface ApiKeyEntry {
  key: string;
  label: string;
  enabled: boolean;
  usage: KeyUsage;
}

export interface ModelEntry {
  id: string;
  label: string;
  enabled: boolean;
}

export interface WarpExitEntry {
  idx: number;
  ready: boolean;
  status: string;
  reason: string;
  socks: number;
  registered: boolean;
  error: string;
}

export interface ProviderEntry {
  id: string;
  label: string;
  kind: string;
  slots: number;
  models: string[];
  enabled: boolean;
  deletable: boolean;
  retry_in: number;
  retry_reason: string;
  health: {
    active: number;
    fetched_at: number;
    error: string;
    exits: WarpExitEntry[];
  };
}

export interface ReconnectResult {
  id: string;
  ok: boolean;
  pool: Record<string, unknown>;
  before: { active: number; ready: number; exits: number; error: string };
  after: { active: number; ready: number; exits: number; error: string };
}

export interface RecentEntry {
  ts: number;
  model: string;
  status: number;
  ms: number;
  warp_idx: number | null;
  error: string;
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, { credentials: "same-origin", ...init });
  if (r.status === 401) throw new ApiError(401, "admin login required");
  if (!r.ok) {
    let detail = r.statusText;
    try {
      const body = await r.json();
      detail = body?.detail ?? body?.error?.message ?? detail;
    } catch {
      /* keep statusText */
    }
    throw new ApiError(r.status, String(detail));
  }
  return (await r.json()) as T;
}

const json = (body: unknown) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

/** Generic CRUD client for one admin collection: list/create/update/delete. */
function resource<List, Entry, Create, Patch>(base: string) {
  const url = (id?: string) =>
    id === undefined ? base : `${base}/${encodeURIComponent(id)}`;
  return {
    list: () => req<List>(url()),
    create: (body: Create) => req<Entry>(url(), json(body)),
    update: (id: string, body: Patch) =>
      req<Entry>(url(id), { ...json(body), method: "PUT" }),
    remove: (id: string) => req<{ status: string }>(url(id), { method: "DELETE" }),
  };
}

const keysRes = resource<{ keys: ApiKeyEntry[] }, ApiKeyEntry,
  { key: string; label: string; enabled: boolean },
  { label?: string; enabled?: boolean }>("/api/admin/keys");
const modelsRes = resource<{ models: ModelEntry[] }, ModelEntry,
  { id: string; label: string; enabled: boolean },
  { label?: string; enabled?: boolean }>("/api/admin/models");
const providersRes = resource<{ providers: ProviderEntry[] }, { id: string },
  {
    id: string;
    label: string;
    kind: string;
    slots: number;
    models: string[];
    enabled: boolean;
  },
  { label?: string; slots?: number; models?: string[]; enabled?: boolean }
>("/api/admin/providers");

export const api = {
  keys: keysRes.list,
  createKey: keysRes.create,
  updateKey: keysRes.update,
  deleteKey: keysRes.remove,
  rotateKey: (key: string, new_key?: string) =>
    req<ApiKeyEntry>(`/api/admin/keys/${encodeURIComponent(key)}/rotate`, {
      ...json({ new_key: new_key ?? "" }),
    }),
  models: modelsRes.list,
  createModel: modelsRes.create,
  updateModel: (id: string, body: { label?: string; enabled?: boolean }) =>
    req<ModelEntry>(`/api/admin/models/${encodeURIComponent(id)}`, {
      ...json(body),
      method: "PUT",
    }),
  deleteModel: modelsRes.remove,
  providers: providersRes.list,
  createProvider: providersRes.create,
  updateProvider: providersRes.update,
  deleteProvider: providersRes.remove,
  providerHealth: (id: string) =>
    req<ProviderEntry & { debug: Record<string, unknown> }>(
      `/api/admin/providers/${encodeURIComponent(id)}/health`,
    ),
  reconnectProvider: (id: string) =>
    req<ReconnectResult>(
      `/api/admin/providers/${encodeURIComponent(id)}/reconnect`,
      { method: "POST" },
    ),
  providerRecent: (id: string) =>
    req<{ recent: RecentEntry[] }>(
      `/api/admin/providers/${encodeURIComponent(id)}/recent`,
    ),
  logout: () => req<{ status: string }>("/api/admin/logout", { method: "POST" }),
};
