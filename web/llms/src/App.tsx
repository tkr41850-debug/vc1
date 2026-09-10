import { useCallback, useEffect, useState } from "react";
import { ApiError, api, type ApiKeyEntry, type ModelEntry } from "./api";
import KeysTab from "./components/KeysTab";
import ModelsTab from "./components/ModelsTab";

type Tab = "keys" | "models";

export default function App() {
  const [tab, setTab] = useState<Tab>("keys");
  const [keys, setKeys] = useState<ApiKeyEntry[]>([]);
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [authed, setAuthed] = useState(true);
  const [loading, setLoading] = useState(true);

  const onAuthError = useCallback(() => setAuthed(false), []);

  const reload = useCallback(async () => {
    try {
      const [k, m] = await Promise.all([api.keys(), api.models()]);
      setKeys(k.keys);
      setModels(m.models);
      setAuthed(true);
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) setAuthed(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    reload();
    const t = setInterval(reload, 15000);
    return () => clearInterval(t);
  }, [reload]);

  const logout = async () => {
    await api.logout();
    setAuthed(false);
  };

  if (loading) {
    return <div className="p-8 text-gray-500">Loading…</div>;
  }

  if (!authed) {
    return (
      <div className="mx-auto mt-24 max-w-md rounded border p-8 text-center">
        <h1 className="mb-2 text-xl font-semibold">llms admin</h1>
        <p className="mb-4 text-sm text-gray-600">Sign in with GitHub to manage keys and models.</p>
        <a
          className="inline-block rounded bg-gray-900 px-4 py-2 text-sm text-white hover:bg-gray-700"
          href="/api/admin/login"
        >
          Sign in with GitHub
        </a>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-5xl p-6">
      <header className="mb-4 flex items-center justify-between">
        <h1 className="text-xl font-semibold">llms admin</h1>
        <button className="text-sm text-gray-600 hover:underline" onClick={logout}>
          Sign out
        </button>
      </header>
      <nav className="mb-4 flex gap-2 border-b">
        {(["keys", "models"] as Tab[]).map((t) => (
          <button
            key={t}
            className={`px-3 py-2 text-sm capitalize ${
              tab === t
                ? "border-b-2 border-blue-600 font-medium"
                : "text-gray-500 hover:text-gray-800"
            }`}
            onClick={() => setTab(t)}
          >
            {t}
          </button>
        ))}
      </nav>
      {tab === "keys" ? (
        <KeysTab keys={keys} reload={reload} onAuthError={onAuthError} />
      ) : (
        <ModelsTab models={models} reload={reload} onAuthError={onAuthError} />
      )}
    </div>
  );
}
