import { useCallback, useEffect, useState } from "react";
import { ApiError, api, type ApiKeyEntry, type ModelEntry, type ProviderEntry } from "./api";
import KeysTab from "./components/KeysTab";
import ModelsTab from "./components/ModelsTab";
import ProvidersTab from "./components/ProvidersTab";

type Tab = "keys" | "models" | "providers";

const TABS: Tab[] = ["keys", "models", "providers"];

function tabFromPath(pathname: string): Tab {
  const segs = pathname.split("/").filter(Boolean);
  const last = segs[segs.length - 1] ?? "";
  return (TABS as string[]).includes(last) ? (last as Tab) : "keys";
}

export default function App() {
  const [tab, setTab] = useState<Tab>(() => tabFromPath(window.location.pathname));
  const [keys, setKeys] = useState<ApiKeyEntry[]>([]);
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [providers, setProviders] = useState<ProviderEntry[]>([]);
  const [authed, setAuthed] = useState(true);
  const [loading, setLoading] = useState(true);
  const [live, setLive] = useState(true);

  const onAuthError = useCallback(() => setAuthed(false), []);

  const navigate = useCallback((t: Tab) => {
    setTab(t);
    window.history.pushState(null, "", `/ui/${t}`);
  }, []);

  // Collection data flows ONLY over SSE (/api/admin/*/sse): snapshot on
  // connect, then pushes on every CRUD write. No GET polling — mutations
  // rely on the hub push, so this is a stable no-op kept for the tab props.
  const reload = useCallback(() => {}, []);

  useEffect(() => {
    const onPop = () => setTab(tabFromPath(window.location.pathname));
    window.addEventListener("popstate", onPop);
    // EventSource auto-reconnects (snapshot on connect), so nothing needs
    // refetching on visibility change or on transient errors.
    let snapshots = 0;
    let failures = 0;
    let authProbed = false;
    const firstSnapshot = () => {
      snapshots += 1;
      setAuthed(true);
      setLoading(false);
      setLive(true);
    };
    const streamFailed = () => {
      failures += 1;
      if (snapshots > 0) {
        // Data was flowing: transient drop, EventSource retries itself.
        setLive(false);
        return;
      }
      if (failures >= 3 && !authProbed) {
        authProbed = true;
        // EventSource hides the status code, so a single probe tells a
        // logged-out session (login screen) from a down server (banner).
        // Failure-path only — never steady-state polling.
        api.keys().then(
          () => {
            setLoading(false);
            setLive(false);
          },
          (e) => {
            if (e instanceof ApiError && e.status === 401) {
              setAuthed(false);
              setLoading(false);
            } else {
              setLoading(false);
              setLive(false);
            }
          },
        );
      }
    };
    const subs: EventSource[] = [];
    const watch = (
      url: string,
      apply: (msg: { keys?: ApiKeyEntry[]; models?: ModelEntry[]; providers?: ProviderEntry[] }) => void,
    ) => {
      let gotData = false;
      const es = new EventSource(url);
      subs.push(es);
      es.onmessage = (ev) => {
        try {
          apply(JSON.parse(ev.data));
        } catch {
          /* keep old */
        }
        if (!gotData) {
          gotData = true;
          firstSnapshot();
        } else {
          setLive(true);
        }
      };
      es.onerror = () => {
        if (es.readyState === EventSource.CLOSED) return;
        if (!gotData) streamFailed();
        else setLive(false);
      };
    };
    watch("/api/admin/keys/sse", (msg) => {
      if (msg.keys) setKeys(msg.keys);
    });
    watch("/api/admin/models/sse", (msg) => {
      if (msg.models) setModels(msg.models);
    });
    watch("/api/admin/providers/sse", (msg) => {
      if (msg.providers) setProviders(msg.providers);
    });
    return () => {
      window.removeEventListener("popstate", onPop);
      subs.forEach((es) => es.close());
    };
  }, []);

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
        <div className="flex items-center gap-3">
          {!live && <span className="text-xs text-amber-600">reconnecting…</span>}
          <button className="text-sm text-gray-600 hover:underline" onClick={logout}>
            Sign out
          </button>
        </div>
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
            onClick={() => navigate(t)}
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
