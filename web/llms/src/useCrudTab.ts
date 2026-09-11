import { useState } from "react";
import { ApiError } from "./api";

/** Shared add/toggle/remove behavior for the admin CRUD tabs. */
export function useCrudTab(onAuthError: () => void) {
  const [error, setError] = useState("");

  const fail = (e: unknown) => {
    if (e instanceof ApiError && e.status === 401) return onAuthError();
    setError(e instanceof Error ? e.message : String(e));
  };

  const run = async (fn: () => Promise<unknown>, reload: () => void) => {
    setError("");
    try {
      await fn();
      reload();
    } catch (e) {
      fail(e);
    }
  };

  return { error, setError, run };
}
