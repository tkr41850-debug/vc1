import { useState } from "react";
import { ApiError, api, type ApiKeyEntry } from "../api";

function fmt(n: number): string {
  return n.toLocaleString("en-US");
}

function mask(secret: string): string {
  if (secret.length <= 8) return "••••";
  return `${secret.slice(0, 6)}…${secret.slice(-4)}`;
}

function SecretCell({ value }: { value: string }) {
  const [revealed, setRevealed] = useState(false);
  const shown = revealed ? value : mask(value);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(value);
    } catch {
      /* clipboard unavailable; reveal still works */
    }
  };
  return (
    <span className="inline-flex items-center gap-1">
      <span title={revealed ? undefined : "Click reveal to show full secret"}>{shown}</span>
      <button
        className="rounded px-1 text-xs text-gray-500 hover:bg-gray-100"
        onClick={() => setRevealed((v) => !v)}
        title={revealed ? "Mask" : "Reveal"}
      >
        {revealed ? "🙈" : "👁"}
      </button>
      <button
        className="rounded px-1 text-xs text-gray-500 hover:bg-gray-100"
        onClick={copy}
        title="Copy full secret"
      >
        📋
      </button>
    </span>
  );
}

export default function KeysTab({
  keys,
  reload,
  onAuthError,
}: {
  keys: ApiKeyEntry[];
  reload: () => void;
  onAuthError: () => void;
}) {
  const [form, setForm] = useState({ key: "", label: "" });
  const [error, setError] = useState("");

  const fail = (e: unknown) => {
    if (e instanceof ApiError && e.status === 401) return onAuthError();
    setError(e instanceof Error ? e.message : String(e));
  };

  const add = async () => {
    if (!form.key.trim()) {
      setError("key is required");
      return;
    }
    if (!form.key.trim().startsWith("sk-")) {
      setError("secret keys must start with sk- (ak- is affinity, not auth)");
      return;
    }
    setError("");
    try {
      await api.createKey({ key: form.key.trim(), label: form.label, enabled: true });
      setForm({ key: "", label: "" });
      reload();
    } catch (e) {
      fail(e);
    }
  };

  const toggle = async (k: ApiKeyEntry) => {
    setError("");
    try {
      await api.updateKey(k.key, { enabled: !k.enabled });
      reload();
    } catch (e) {
      fail(e);
    }
  };

  const remove = async (k: ApiKeyEntry) => {
    if (!window.confirm(`Delete key ${k.key}? This orphans its usage history.`)) return;
    setError("");
    try {
      await api.deleteKey(k.key);
      reload();
    } catch (e) {
      fail(e);
    }
  };

  const rotate = async (k: ApiKeyEntry) => {
    if (!window.confirm(`Rotate ${k.key}? A new secret is issued; the old one is disabled and usage carries over.`)) return;
    setError("");
    try {
      await api.rotateKey(k.key);
      reload();
    } catch (e) {
      fail(e);
    }
  };

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-end gap-2">
        <label className="flex flex-col text-sm">
          Secret key
          <input
            className="rounded border px-2 py-1 font-mono"
            placeholder="sk-team1"
            value={form.key}
            onChange={(e) => setForm({ ...form, key: e.target.value })}
            title="sk- secret, sent on the Authorization header. ak- is affinity, not auth."
          />
        </label>
        <label className="flex flex-col text-sm">
          Label
          <input
            className="rounded border px-2 py-1"
            placeholder="Team 1"
            value={form.label}
            onChange={(e) => setForm({ ...form, label: e.target.value })}
          />
        </label>
        <button
          className="rounded bg-blue-600 px-3 py-1.5 text-sm text-white hover:bg-blue-700"
          onClick={add}
        >
          Add key
        </button>
        {error && <span className="text-sm text-red-600">{error}</span>}
      </div>
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b text-left text-gray-600">
            <th className="py-2 pr-4">Key</th>
            <th className="py-2 pr-4">Label</th>
            <th className="py-2 pr-4">Enabled</th>
            <th className="py-2 pr-4 text-right">Requests</th>
            <th className="py-2 pr-4 text-right">In</th>
            <th className="py-2 pr-4 text-right">Cached</th>
            <th className="py-2 pr-4 text-right">Uncached</th>
            <th className="py-2 pr-4 text-right">Out</th>
            <th className="py-2 pr-4 text-right">Reasoning</th>
            <th className="py-2"></th>
          </tr>
        </thead>
        <tbody>
          {keys.map((k) => (
            <tr key={k.key} className="border-b hover:bg-gray-50">
              <td className="py-2 pr-4 font-mono">
                <SecretCell value={k.key} />
              </td>
              <td className="py-2 pr-4">{k.label || "—"}</td>
              <td className="py-2 pr-4">
                <button
                  className={`rounded px-2 py-0.5 text-xs ${
                    k.enabled
                      ? "bg-green-100 text-green-800"
                      : "bg-gray-200 text-gray-600"
                  }`}
                  onClick={() => toggle(k)}
                  title="Toggle enabled"
                >
                  {k.enabled ? "on" : "off"}
                </button>
              </td>
              <td className="py-2 pr-4 text-right">{fmt(k.usage.requests)}</td>
              <td className="py-2 pr-4 text-right">{fmt(k.usage.input_tokens)}</td>
              <td className="py-2 pr-4 text-right" title="Served from prompt cache">
                {fmt(k.usage.cached_tokens ?? 0)}
              </td>
              <td className="py-2 pr-4 text-right" title="Billed input tokens">
                {fmt(k.usage.input_tokens - (k.usage.cached_tokens ?? 0))}
              </td>
              <td className="py-2 pr-4 text-right">{fmt(k.usage.output_tokens)}</td>
              <td className="py-2 pr-4 text-right">{fmt(k.usage.reasoning_tokens ?? 0)}</td>
              <td className="py-2 text-right">
                <button
                  className="mr-1 rounded px-2 py-0.5 text-xs text-blue-600 hover:bg-blue-50"
                  onClick={() => rotate(k)}
                  title="Issue a new secret; old disabled, usage carries over"
                >
                  rotate
                </button>
                <button
                  className="rounded px-2 py-0.5 text-xs text-red-600 hover:bg-red-50"
                  onClick={() => remove(k)}
                >
                  delete
                </button>
              </td>
            </tr>
          ))}
          {keys.length === 0 && (
            <tr>
              <td colSpan={10} className="py-4 text-center text-gray-500">
                No keys yet — add one above.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
