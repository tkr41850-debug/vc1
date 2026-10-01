import { useCallback, useEffect, useRef, useState } from "react";
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
  const [hasData, setHasData] = useState(false);
  const subsRef = useRef<EventSource[]>([]);

  const onAuthError = useCallback(() => setAuthed(false), []);

  const navigate = useCallback((t: Tab) => {
    setTab(t);
    window.history.pushState(null, "", `/ui/${t}`);
  }, []);

  // Collection data flows ONLY over SSE (/api/admin/*/sse): snapshot on
  // connect, then pushes on every CRUD write. No GET polling — mutations
  // apply the returned transition snapshot locally; reload stays a
  // stable no-op kept for the tab props.
  const reload = useCallback(() => {}, []);

  const applyProviders = useCallback(
    (list: ProviderEntry[] | ((prev: ProviderEntry[]) => ProviderEntry[])) => {
      // Cheap merge: same order + same ids update in place so the table
      // does not remount on every throttled Busy tick. Accepts a functional
      // updater so optimistic acks merge onto current state, never a stale
      // closure over the render-time list.
      const merge = (prev: ProviderEntry[], next: ProviderEntry[]) => {
        if (
          prev.length === next.length &&
          prev.every((p, i) => p.id === next[i].id)
        ) {
          let same = true;
          const merged = prev.map((p, i) => {
            const n = next[i];
          if (
            p.lifecycle !== n.lifecycle ||
            p.in_flight !== n.in_flight ||
            p.enabled !== n.enabled ||
            p.retry_in !== n.retry_in ||
            p.retry_reason !== n.retry_reason ||
            p.cycling !== n.cycling ||
            p.cycle_cooldown_remaining !== n.cycle_cooldown_remaining ||
            p.label !== n.label ||
            p.kind !== n.kind ||
            p.deletable !== n.deletable ||
            p.exits !== n.exits ||
            p.models.join("\n") !== n.models.join("\n") ||
            p.drain?.until_ms !== n.drain?.until_ms ||
            p.drain?.forced !== n.drain?.forced ||
            p.health.fetched_at !== n.health.fetched_at ||
            p.health.error !== n.health.error ||
            p.health.exits.length !== n.health.exits.length ||
            p.health.exits.some(
              (w, j) =>
                w.ready !== n.health.exits[j]?.ready ||
                w.status !== n.health.exits[j]?.status ||
                w.reason !== n.health.exits[j]?.reason ||
                w.socks !== n.health.exits[j]?.socks ||
                w.registered !== n.health.exits[j]?.registered ||
                w.error !== n.health.exits[j]?.error,
            )
          ) {
            same = false;
            return n;
          }
          return p;
        });
        return same ? prev : merged;
      }
      return next;
      };
      if (typeof list === "function") setProviders((prev) => merge(prev, list(prev)));
      else setProviders((prev) => merge(prev, list));
    },
    [],
  );

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } catch {
      /* expired session already logged out server-side; fall through */
    } finally {
      // Stop the streams: otherwise they 401-retry forever behind the
      // login screen (App never unmounts).
      subsRef.current.forEach((es) => es.close());
      subsRef.current = [];
      setAuthed(false);
    }
  }, []);

  useEffect(() => {
    const onPop = () => setTab(tabFromPath(window.location.pathname));
    window.addEventListener("popstate", onPop);
    // EventSource auto-reconnects (snapshot on connect), so nothing needs
    // refetching on visibility change or on transient errors.
    let snapshots = 0;
    let failures = 0;
    let postFails = 0;
    let reProbeAt = 0;
    let authProbed = false;
    const closeSubs = () => {
      subsRef.current.forEach((es) => es.close());
      subsRef.current = [];
    };
    const goLoggedOut = () => {
      closeSubs();
      setAuthed(false);
      setLoading(false);
    };
    // Single failure-path probe (never steady-state polling): EventSource
    // hides the status code, so this tells logged-out (login screen) from
    // a down server (banner).
    const probeAuth = () => {
      authProbed = true;
      api.keys().then(
        () => {
          setLoading(false);
          setLive(false);
        },
        (e) => {
          if (e instanceof ApiError && e.status === 401) goLoggedOut();
          else {
            setLoading(false);
            setLive(false);
          }
        },
      );
    };
    const firstSnapshot = () => {
      snapshots += 1;
      postFails = 0;
      setAuthed(true);
      setHasData(true);
      // All three collections clear the spinner; a 5s fallback covers a
      // hung third stream so partial data still renders (see below).
      if (snapshots >= 3) setLoading(false);
      setLive(true);
    };
    const streamFailed = () => {
      failures += 1;
      if (snapshots > 0) {
        // Data was flowing: transient drop, EventSource retries itself.
        // But a logged-out session 401-loops forever the same way, so
        // re-probe after sustained darkness (6 straight errors ≈ 2 per
        // stream) instead of sitting on the banner indefinitely.
        postFails += 1;
        setLive(false);
        if (postFails >= 6 && failures >= reProbeAt) {
          reProbeAt = failures + 6;
          authProbed = false;
          probeAuth();
        }
        return;
      }
      if (failures >= 3 && !authProbed) probeAuth();
    };
    const subs = subsRef.current;
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
        postFails = 0;
        if (!gotData) {
          gotData = true;
          firstSnapshot();
        } else {
          setLive(true);
        }
      };
      es.onerror = () => {
        if (es.readyState === EventSource.CLOSED) return;
        streamFailed();
      };
    };
    watch("/api/admin/keys/sse", (msg) => {
      if (msg.keys) setKeys(msg.keys);
    });
    watch("/api/admin/models/sse", (msg) => {
      if (msg.models) setModels(msg.models);
    });
    watch("/api/admin/providers/sse", (msg) => {
      if (msg.providers) applyProviders(msg.providers);
    });
    // Fallback so one hung stream can't hold the spinner forever; the
    // arrived slices render with the reconnecting banner until it lands.
    const t = setTimeout(() => setLoading(false), 5000);
    return () => {
      window.removeEventListener("popstate", onPop);
      clearTimeout(t);
      subs.forEach((es) => es.close());
      subsRef.current = [];
    };
  }, []);

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
          {!live && hasData && <span className="text-xs text-amber-600">reconnecting…</span>}
          {!live && !hasData && (
            <span className="text-xs text-amber-600">server unreachable — retrying…</span>
          )}
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
        <ProvidersTab
          providers={providers}
          reload={reload}
          onAuthError={onAuthError}
          onProviders={applyProviders}
        />
      )}
    </div>
  );
}
