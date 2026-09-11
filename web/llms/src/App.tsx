import { useCallback, useEffect, useState } from "react";
import { ApiError, api, type ApiKeyEntry, type ModelEntry, type ProviderEntry } from "./api";
import KeysTab from "./components/KeysTab";
import ModelsTab from "./components/ModelsTab";
import ProvidersTab from "./components/ProvidersTab";

type Tab = "keys" | "models" | "providers";

export default function App() {
  const [tab, setTab] = useState<Tab>("keys");
  const [keys, setKeys] = useState<ApiKeyEntry[]>([]);
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [providers, setProviders] = useState<ProviderEntry[]>([]);
  const [authed, setAuthed] = useState(true);
  const [loading, setLoading] = useState(true);

  const onAuthError = useCallback(() => setAuthed(false), []);

  const reload = useCallback(async () => {
    try {
      const [k, m, p] = await Promise.all([api.keys(), api.models(), api.providers()]);
      setKeys(k.keys);
      setModels(m.models);
      setProviders(p.providers);
      setAuthed(true);
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) setAuthed(false);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    reload();
    // Pause background polling while the tab is hidden: an idle admin page
    // shouldn't keep churning the server's YAML parses + usage snapshots.
    const onVis = () => {
      if (!document.hidden) reload();
    };
    document.addEventListener("visibilitychange", onVis);
    const t = setInterval(() => {
      if (!document.hidden) reload();
    }, 15000);
    return () => {
      document.removeEventListener("visibilitychange", onVis);
      clearInterval(t);
    };
  }, [reload]);

  const logout = async () => {
    try {
      await api.logout();
    } catch {
      /* expired session already logged out server-side; fall through */
    } finally {
      setAuthed(false);
    }
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
        {(["keys", "models", "providers"] as Tab[]).map((t) => (
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
      ) : tab === "models" ? (
        <ModelsTab models={models} reload={reload} onAuthError={onAuthError} />
      ) : (
        <ProvidersTab providers={providers} reload={reload} onAuthError={onAuthError} />
      )}
    </div>
  );
}
