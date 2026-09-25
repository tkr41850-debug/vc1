import { useEffect, useRef, useState } from "react";
import { ApiError, api, type ProviderEntry, type RecentEntry } from "../api";
import { useCrudTab } from "../useCrudTab";

function fmtRetry(sec: number): string {
  if (sec <= 0) return "—";
  if (sec < 60) return `${Math.ceil(sec)}s`;
  return `${Math.floor(sec / 60)}m ${Math.ceil(sec % 60)}s`;
}

function HealthDot({ p }: { p: ProviderEntry }) {
  if (!p.enabled) return <span title="disabled">⚪</span>;
  if (p.kind === "noproxy") return <span title="direct">🟢</span>;
  if (p.retry_in > 0) return <span title={`rate limited: ${p.retry_reason}`}>🟡</span>;
  const ready = p.health.exits.filter((w) => w.ready).length;
  if (p.health.error) return <span title={p.health.error}>🔴</span>;
  if (ready === 0) return <span title="no ready exits">🔴</span>;
  return <span title={`${ready} ready exits`}>🟢</span>;
}

function DebugModal({
  provider,
  onClose,
  onAuthError,
}: {
  provider: ProviderEntry;
  onClose: () => void;
  onAuthError: () => void;
}) {
  const [live, setLive] = useState<ProviderEntry | null>(null);
  const [recent, setRecent] = useState<RecentEntry[]>([]);
  const [debug, setDebug] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");
  const [streamError, setStreamError] = useState("");
  const [notice, setNotice] = useState("");
  const [reconnecting, setReconnecting] = useState(false);
  const [modelsText, setModelsText] = useState(() => provider.models.join("\n"));
  const [savingModels, setSavingModels] = useState(false);
  const esRef = useRef<EventSource | null>(null);

  const shown = live ?? provider;

  const refresh = () => {
    api
      .providerHealth(provider.id)
      .then((h) => {
        setLive(h);
        setDebug((h.debug as Record<string, unknown>) ?? null);
      })
      .catch((e) => {
        if (e instanceof ApiError && e.status === 401) return onAuthError();
        setError(e instanceof Error ? e.message : String(e));
      });
  };

  const reconnect = () => {
    setReconnecting(true);
    setNotice("");
    api
      .reconnectProvider(provider.id)
      .then((r) => {
        setNotice(
          `reconnected: ${r.after.ready}/${r.after.exits} exits ready (was ${r.before.ready}/${r.before.exits})`,
        );
        refresh();
      })
      .catch((e) => {
        if (e instanceof ApiError && e.status === 401) return onAuthError();
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => setReconnecting(false));
  };

  const saveModels = () => {
    const models = modelsText
      .split("\n")
      .map((m) => m.trim())
      .filter(Boolean);
    setSavingModels(true);
    setNotice("");
    api
      .updateProvider(provider.id, { models })
      .then(() => {
        setNotice(`saved ${models.length} model pattern(s)`);
        refresh();
      })
      .catch((e) => {
        if (e instanceof ApiError && e.status === 401) return onAuthError();
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => setSavingModels(false));
  };

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  useEffect(() => {
    let cancelled = false;
    api
      .providerHealth(provider.id)
      .then((h) => {
        if (!cancelled) {
          setLive(h);
          setDebug((h.debug as Record<string, unknown>) ?? null);
        }
      })
      .catch((e) => {
        if (e instanceof ApiError && e.status === 401) return onAuthError();
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    api
      .providerRecent(provider.id)
      .then((r) => {
        if (!cancelled) setRecent(r.recent);
      })
      .catch(() => {});
    const es = new EventSource(`/api/admin/providers/${encodeURIComponent(provider.id)}/stream`);
    esRef.current = es;
    es.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.recent) setRecent(msg.recent);
        else if (msg.request) setRecent((prev) => [...prev.slice(-9), msg.request]);
      } catch {
        /* keep old */
      }
    };
    es.onerror = () => {
      if (!cancelled) {
        setStreamError("Live stream disconnected — recent requests may be stale.");
      }
    };
    return () => {
      cancelled = true;
      es.close();
      esRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider.id]);

  return (
    <div
      className="fixed inset-0 z-10 flex items-center justify-center bg-black/40"
      onClick={onClose}
    >
      <div
        className="max-h-[85vh] w-full max-w-3xl overflow-auto rounded bg-white p-6"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-3 flex items-center justify-between">
          <h2 className="font-mono text-lg">{provider.id}</h2>
          <div className="flex items-center gap-2">
            {provider.kind === "warp" && (
              <button
                className="rounded bg-blue-600 px-2 py-1 text-xs text-white hover:bg-blue-700 disabled:opacity-50"
                onClick={reconnect}
                disabled={reconnecting}
                title="Bounce the pool, drop cached egress, re-poll health"
              >
                {reconnecting ? "reconnecting…" : "reconnect"}
              </button>
            )}
            <button className="text-sm text-gray-600 hover:underline" onClick={onClose}>
              Close (esc)
            </button>
          </div>
        </div>
        {error && <p className="mb-2 text-sm text-red-600">{error}</p>}
        {notice && <p className="mb-2 text-sm text-green-700">{notice}</p>}
        {streamError && <p className="mb-2 text-sm text-amber-600">{streamError}</p>}
        <h3 className="mb-1 text-sm font-medium text-gray-600">Warp exits</h3>
        <table className="mb-4 w-full border-collapse text-sm">
          <thead>
            <tr className="border-b text-left text-gray-600">
              <th className="py-1 pr-3">#</th>
              <th className="py-1 pr-3">Ready</th>
              <th className="py-1 pr-3">Status</th>
              <th className="py-1 pr-3">Socks</th>
              <th className="py-1 pr-3">Reg</th>
              <th className="py-1">Error</th>
            </tr>
          </thead>
          <tbody>
            {shown.health.exits.map((w) => (
              <tr key={w.idx} className="border-b font-mono text-xs">
                <td className="py-1 pr-3">{w.idx}</td>
                <td className="py-1 pr-3">{w.ready ? "yes" : "no"}</td>
                <td className="py-1 pr-3">{w.status || "—"}</td>
                <td className="py-1 pr-3">{w.socks || "—"}</td>
                <td className="py-1 pr-3">{w.registered ? "yes" : "no"}</td>
                <td className="py-1">{w.error || w.reason || "—"}</td>
              </tr>
            ))}
            {shown.health.exits.length === 0 && (
              <tr>
                <td colSpan={6} className="py-2 text-center text-gray-500">
                  {provider.kind === "noproxy"
                    ? "Direct egress has no warp exits."
                    : "No health data yet."}
                </td>
              </tr>
            )}
          </tbody>
        </table>
        <div className="mb-4">
          <h3 className="mb-1 text-sm font-medium text-gray-600">
            Models (one pattern per line, * = prefix)
          </h3>
          <textarea
            className="mb-2 w-full rounded border px-2 py-1 font-mono text-xs"
            rows={Math.min(8, Math.max(2, shown.models.length + 1))}
            value={modelsText}
            onChange={(e) => setModelsText(e.target.value)}
          />
          <button
            className="rounded bg-blue-600 px-2 py-1 text-xs text-white hover:bg-blue-700 disabled:opacity-50"
            onClick={saveModels}
            disabled={savingModels}
          >
            {savingModels ? "saving…" : "save models"}
          </button>
        </div>
        {debug && Object.keys(debug).length > 0 && (
          <>
            <h3 className="mb-1 text-sm font-medium text-gray-600">Pool debug</h3>
            <pre className="mb-4 max-h-48 overflow-auto rounded bg-gray-50 p-2 font-mono text-xs">
              {JSON.stringify(debug, null, 2)}
            </pre>
          </>
        )}
        <h3 className="mb-1 text-sm font-medium text-gray-600">
          Last requests (live)
        </h3>
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="border-b text-left text-gray-600">
              <th className="py-1 pr-3">Time</th>
              <th className="py-1 pr-3">Model</th>
              <th className="py-1 pr-3">Status</th>
              <th className="py-1 pr-3">ms</th>
              <th className="py-1 pr-3">Warp</th>
              <th className="py-1">Error</th>
            </tr>
          </thead>
          <tbody>
            {[...recent].reverse().map((r, i) => (
              <tr key={`${r.ts}-${i}`} className="border-b font-mono text-xs">
                <td className="py-1 pr-3">
                  {new Date(r.ts * 1000).toLocaleTimeString()}
                </td>
                <td className="py-1 pr-3">{r.model}</td>
                <td className="py-1 pr-3">{r.status}</td>
                <td className="py-1 pr-3">{r.ms}</td>
                <td className="py-1 pr-3">{r.warp_idx ?? "—"}</td>
                <td className="py-1">{r.error || "—"}</td>
              </tr>
            ))}
            {recent.length === 0 && (
              <tr>
                <td colSpan={6} className="py-2 text-center text-gray-500">
                  No requests yet — leave this open while traffic flows.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function ProvidersTab({
  providers,
  reload,
  onAuthError,
}: {
  providers: ProviderEntry[];
  reload: () => void;
  onAuthError: () => void;
}) {
  const [form, setForm] = useState({ id: "", label: "", models: "", exits: "1" });
  const [debugId, setDebugId] = useState<string | null>(null);
  const { error, setError, run } = useCrudTab(onAuthError);

  const add = async () => {
    if (!form.id.trim()) {
      setError("id is required");
      return;
    }
    const exits = Math.max(1, parseInt(form.exits, 10) || 1);
    await run(async () => {
      await api.createProvider({
        id: form.id.trim(),
        label: form.label,
        kind: "warp",
        models: form.models.split(",").map((m) => m.trim()).filter(Boolean),
        enabled: true,
        exits,
      });
      setForm({ id: "", label: "", models: "", exits: "1" });
    }, reload);
  };

  const toggle = (p: ProviderEntry) =>
    run(() => api.updateProvider(p.id, { enabled: !p.enabled }), reload);

  const remove = (p: ProviderEntry) => {
    if (!window.confirm(`Delete provider ${p.id}?`)) return;
    return run(() => api.deleteProvider(p.id), reload);
  };

  const debugProvider = debugId ? providers.find((p) => p.id === debugId) ?? null : null;

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-end gap-2">
        <label className="flex flex-col text-sm">
          Id
          <input
            className="rounded border px-2 py-1 font-mono"
            placeholder="warp-1"
            value={form.id}
            onChange={(e) => setForm({ ...form, id: e.target.value })}
          />
        </label>
        <label className="flex flex-col text-sm">
          Label
          <input
            className="rounded border px-2 py-1"
            placeholder="Hetzner pool"
            value={form.label}
            onChange={(e) => setForm({ ...form, label: e.target.value })}
          />
        </label>
        <label className="flex flex-col text-sm">
          Exits
          <input
            className="rounded border px-2 py-1 font-mono"
            placeholder="1"
            value={form.exits}
            onChange={(e) => setForm({ ...form, exits: e.target.value })}
          />
        </label>
        <label className="flex flex-col text-sm">
          Models (comma, * = prefix)
          <input
            className="rounded border px-2 py-1 font-mono"
            placeholder="gpt-*, claude-*"
            value={form.models}
            onChange={(e) => setForm({ ...form, models: e.target.value })}
          />
        </label>
        <button
          className="rounded bg-blue-600 px-3 py-1.5 text-sm text-white hover:bg-blue-700"
          onClick={add}
        >
          Add warp provider
        </button>
        {error && <span className="text-sm text-red-600">{error}</span>}
      </div>
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b text-left text-gray-600">
            <th className="py-2 pr-4"></th>
            <th className="py-2 pr-4">Provider</th>
            <th className="py-2 pr-4">Kind</th>
            <th className="py-2 pr-4">Exits</th>
            <th className="py-2 pr-4">Models</th>
            <th className="py-2 pr-4">Enabled</th>
            <th className="py-2 pr-4 text-right">RetryIn</th>
            <th className="py-2"></th>
          </tr>
        </thead>
        <tbody>
          {providers.map((p) => (
            <tr key={p.id} className="border-b hover:bg-gray-50">
              <td className="py-2 pr-2">
                <HealthDot p={p} />
              </td>
              <td className="py-2 pr-4">
                <span className="font-mono">{p.id}</span>
                {p.label && <span className="ml-2 text-gray-500">{p.label}</span>}
              </td>
              <td className="py-2 pr-4">
                <span className="rounded bg-gray-100 px-1.5 py-0.5 font-mono text-xs">
                  {p.kind}
                </span>
              </td>
              <td className="py-2 pr-4 font-mono text-xs">
                {p.kind === "warp" ? p.exits : "—"}
              </td>
              <td className="max-w-xs truncate py-2 pr-4 font-mono text-xs" title={p.models.join(", ")}>
                {p.models.join(", ") || "—"}
              </td>
              <td className="py-2 pr-4">
                <button
                  className={`rounded px-2 py-0.5 text-xs ${
                    p.enabled ? "bg-green-100 text-green-800" : "bg-gray-200 text-gray-600"
                  }`}
                  onClick={() => toggle(p)}
                  title="Toggle enabled"
                >
                  {p.enabled ? "on" : "off"}
                </button>
              </td>
              <td className="py-2 pr-4 text-right font-mono text-xs" title={p.retry_reason || undefined}>
                {fmtRetry(p.retry_in)}
              </td>
              <td className="py-2 text-right">
                <button
                  className="mr-1 rounded px-2 py-0.5 text-xs text-blue-600 hover:bg-blue-50"
                  onClick={() => setDebugId(p.id)}
                >
                  debug
                </button>
                {p.deletable && (
                  <button
                    className="rounded px-2 py-0.5 text-xs text-red-600 hover:bg-red-50"
                    onClick={() => remove(p)}
                  >
                    delete
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {debugProvider && (
        <DebugModal provider={debugProvider} onClose={() => setDebugId(null)} onAuthError={onAuthError} />
      )}
    </div>
  );
}
