import { useRef, useState } from "react";
import { ApiError } from "../api";

/** Click-to-edit label shared by the Keys/Models/Providers tabs.
 *
 *  Renders the value (or a clickable placeholder when empty); clicking
 *  swaps in a text input with ✓ confirm / ✕ cancel buttons. Enter
 *  confirms (async onSave, inline error on failure), Esc cancels and
 *  restores the original. Input and buttons disable while saving; on
 *  success edit mode exits and the parent re-renders the new value.
 *  401s route to onAuthError like useCrudTab; other failures show inline.
 */
export default function EditableLabel({
  value,
  placeholder = "—",
  title,
  className,
  onSave,
  onAuthError,
}: {
  value: string;
  placeholder?: string;
  title?: string;
  className?: string;
  onSave: (newValue: string) => Promise<unknown>;
  onAuthError?: () => void;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState("");
  // Ref guard closes the double-submit window: two synchronous submits
  // (double-Enter, double-click ✓) both see stale `saving === false`
  // before React re-renders. The write is idempotent (same draft), so the
  // state flag alone would be harmless — this just avoids a duplicate PUT.
  const savingRef = useRef(false);

  const start = () => {
    setDraft(value);
    setError("");
    setEditing(true);
  };

  const cancel = () => {
    if (saving) return;
    setEditing(false);
    setDraft("");
    setError("");
  };

  const confirm = async () => {
    if (savingRef.current) return;
    savingRef.current = true;
    setSaving(true);
    setError("");
    try {
      await onSave(draft);
      setEditing(false);
    } catch (e) {
      if (e instanceof ApiError && e.status === 401 && onAuthError) {
        onAuthError();
        setEditing(false);
        return;
      }
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      savingRef.current = false;
      setSaving(false);
    }
  };

  if (!editing) {
    return (
      <button
        className={
          className ??
          (value
            ? "rounded px-1 hover:bg-gray-100"
            : "rounded px-1 text-gray-400 hover:bg-gray-100")
        }
        onClick={start}
        title={title ?? "Click to edit label"}
      >
        {value || placeholder}
      </button>
    );
  }

  // No onBlur handler by design: blur-save would double-fire alongside a
  // ✓ click, and blur-cancel would swallow ✓ clicks (the input blurs
  // before the button's click lands). Explicit ✓/Enter/Esc only.
  return (
    <span className="inline-flex items-center gap-1">
      <input
        className="rounded border px-1 py-0.5 text-sm"
        value={draft}
        autoFocus
        disabled={saving}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.preventDefault();
            void confirm();
          } else if (e.key === "Escape") {
            e.preventDefault();
            cancel();
          }
        }}
      />
      <button
        className="rounded px-1 text-sm text-green-700 hover:bg-green-50 disabled:opacity-50"
        onClick={() => void confirm()}
        disabled={saving}
        title="Save"
      >
        ✓
      </button>
      <button
        className="rounded px-1 text-sm text-gray-500 hover:bg-gray-100 disabled:opacity-50"
        onClick={cancel}
        disabled={saving}
        title="Cancel"
      >
        ✕
      </button>
      {error && <span className="text-xs text-red-600">{error}</span>}
    </span>
  );
}
