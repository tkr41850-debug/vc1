import { useState } from "react";
import { api, type ModelEntry } from "../api";
import { useCrudTab } from "../useCrudTab";

export default function ModelsTab({
  models,
  reload,
  onAuthError,
}: {
  models: ModelEntry[];
  reload: () => void;
  onAuthError: () => void;
}) {
  const [form, setForm] = useState({ id: "", label: "" });
  const { error, setError, run } = useCrudTab(onAuthError);

  const add = async () => {
    if (!form.id.trim()) {
      setError("id is required");
      return;
    }
    await run(async () => {
      await api.createModel({ id: form.id.trim(), label: form.label, enabled: true });
      setForm({ id: "", label: "" });
    }, reload);
  };

  const toggle = (m: ModelEntry) =>
    run(() => api.updateModel(m.id, { enabled: !m.enabled }), reload);

  const remove = (m: ModelEntry) => {
    if (!window.confirm(`Delete model ${m.id}?`)) return;
    return run(() => api.deleteModel(m.id), reload);
  };

  return (
    <div>
      <div className="mb-4 flex flex-wrap items-end gap-2">
        <label className="flex flex-col text-sm">
          Model id
          <input
            className="rounded border px-2 py-1 font-mono"
            placeholder="mimo-v2.5-free"
            value={form.id}
            onChange={(e) => setForm({ ...form, id: e.target.value })}
          />
        </label>
        <label className="flex flex-col text-sm">
          Label
          <input
            className="rounded border px-2 py-1"
            placeholder="Mimo"
            value={form.label}
            onChange={(e) => setForm({ ...form, label: e.target.value })}
          />
        </label>
        <button
          className="rounded bg-blue-600 px-3 py-1.5 text-sm text-white hover:bg-blue-700"
          onClick={add}
        >
          Add model
        </button>
        {error && <span className="text-sm text-red-600">{error}</span>}
      </div>
      <table className="w-full border-collapse text-sm">
        <thead>
          <tr className="border-b text-left text-gray-600">
            <th className="py-2 pr-4">Model</th>
            <th className="py-2 pr-4">Label</th>
            <th className="py-2 pr-4">Enabled</th>
            <th className="py-2"></th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => (
            <tr key={m.id} className="border-b hover:bg-gray-50">
              <td className="py-2 pr-4 font-mono">{m.id}</td>
              <td className="py-2 pr-4">{m.label || "—"}</td>
              <td className="py-2 pr-4">
                <button
                  className={`rounded px-2 py-0.5 text-xs ${
                    m.enabled
                      ? "bg-green-100 text-green-800"
                      : "bg-gray-200 text-gray-600"
                  }`}
                  onClick={() => toggle(m)}
                  title="Toggle enabled"
                >
                  {m.enabled ? "on" : "off"}
                </button>
              </td>
              <td className="py-2 text-right">
                <button
                  className="rounded px-2 py-0.5 text-xs text-red-600 hover:bg-red-50"
                  onClick={() => remove(m)}
                >
                  delete
                </button>
              </td>
            </tr>
          ))}
          {models.length === 0 && (
            <tr>
              <td colSpan={4} className="py-4 text-center text-gray-500">
                No models yet — add one above.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
