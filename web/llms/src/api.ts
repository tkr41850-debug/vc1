export interface KeyUsage {
  requests: number;
  input_tokens: number;
  output_tokens: number;
  models: Record<
    string,
    { requests: number; input_tokens: number; output_tokens: number }
  >;
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
  logout: () => req<{ status: string }>("/api/admin/logout", { method: "POST" }),
};
