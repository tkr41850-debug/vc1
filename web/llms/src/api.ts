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
  base_url: string;
  has_token: boolean;
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

export const api = {
  keys: () => req<{ keys: ApiKeyEntry[] }>("/api/admin/keys"),
  createKey: (body: { key: string; label: string; enabled: boolean }) =>
    req<ApiKeyEntry>("/api/admin/keys", json(body)),
  updateKey: (key: string, body: { label?: string; enabled?: boolean }) =>
    req<ApiKeyEntry>(`/api/admin/keys/${encodeURIComponent(key)}`, {
      ...json(body),
      method: "PUT",
    }),
  deleteKey: (key: string) =>
    req<{ status: string }>(`/api/admin/keys/${encodeURIComponent(key)}`, {
      method: "DELETE",
    }),
  models: () => req<{ models: ModelEntry[] }>("/api/admin/models"),
  createModel: (body: { id: string; label: string; enabled: boolean }) =>
    req<ModelEntry>("/api/admin/models", json(body)),
  updateModel: (id: string, body: { label?: string; enabled?: boolean }) =>
    req<ModelEntry>(`/api/admin/models/${encodeURIComponent(id)}`, {
      ...json(body),
      method: "PUT",
    }),
  deleteModel: (id: string) =>
    req<{ status: string }>(`/api/admin/models/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),
  providers: () => req<{ providers: ProviderEntry[] }>("/api/admin/providers"),
  createProvider: (body: {
    id: string;
    label: string;
    kind: string;
    base_url: string;
    token: string;
    models: string[];
    enabled: boolean;
  }) => req<{ id: string }>("/api/admin/providers", json(body)),
  updateProvider: (
    id: string,
    body: { label?: string; base_url?: string; token?: string; models?: string[]; enabled?: boolean },
  ) =>
    req<{ id: string; enabled: boolean }>(
      `/api/admin/providers/${encodeURIComponent(id)}`,
      { ...json(body), method: "PUT" },
    ),
  deleteProvider: (id: string) =>
    req<{ status: string }>(`/api/admin/providers/${encodeURIComponent(id)}`, {
      method: "DELETE",
    }),
  providerHealth: (id: string) =>
    req<ProviderEntry & { debug: Record<string, unknown> }>(
      `/api/admin/providers/${encodeURIComponent(id)}/health`,
    ),
  providerRecent: (id: string) =>
    req<{ recent: RecentEntry[] }>(
      `/api/admin/providers/${encodeURIComponent(id)}/recent`,
    ),
  logout: () => req<{ status: string }>("/api/admin/logout", { method: "POST" }),
};
