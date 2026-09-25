// KM Tile & Time Split: pure logic (no ComfyUI app/api imports).
// Side-effect free at import time: ComfyUI's frontend loads every .js file in
// WEB_DIRECTORY as an extension module, so this file must only define exports.

export const CUSTOM = "custom";
// Widgets whose values feed the layout (sent to the preview route).
export const PLAN_WIDGETS = [
  "tile_mode", "rows", "cols", "target_width", "target_height", "overlap_mode",
  "overlap", "multiple_of", "chunk_frames", "chunk_overlap", "frame_rule",
];

export function chain(obj, name, fn) {
  const orig = obj[name];
  obj[name] = function (...args) {
    const r = orig?.apply(this, args);
    fn.apply(this, args);
    return r;
  };
}

function widget(node, name) {
  return node.widgets?.find((w) => w.name === name);
}

// Real input widgets: excludes transient/UI widgets (image previews, buttons,
// anything marked non-serializing) that can appear or vanish between runs.
export function isInputWidget(w) {
  return (
    w &&
    w.name &&
    !String(w.name).startsWith("$$") &&
    w.type !== "button" &&
    w.serialize !== false &&
    w.options?.serialize !== false
  );
}

// Small stable string hash (djb2) for the upstream signature.
export function hashString(s) {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
  return (h >>> 0).toString(16);
}

// Id of this node in the API prompt. Nodes inside subgraphs get "parent:child" ids.
export function executionId(output, node) {
  const id = String(node.id);
  if (output[id]?.class_type === "KMTileTimeSplit") return id;
  const hits = Object.keys(output).filter(
    (k) => k.endsWith(":" + id) && output[k].class_type === "KMTileTimeSplit"
  );
  return hits.length === 1 ? hits[0] : null;
}

// Walk input links upstream of `node` inside its own graph, collecting every
// node reached (no dependency on the API-format prompt).
export function collectUpstreamNodes(node) {
  const graph = node.graph;
  const seen = new Set();
  const result = [];
  const stack = [node];
  while (stack.length) {
    const cur = stack.pop();
    for (const input of cur?.inputs ?? []) {
      const linkId = input?.link;
      if (linkId === null || linkId === undefined) continue;
      const link = graph.links?.[linkId] ?? graph.getLink?.(linkId);
      const originId = link?.origin_id;
      if (originId === null || originId === undefined || seen.has(originId)) continue;
      const originNode = graph.getNodeById?.(originId);
      if (!originNode) continue;
      seen.add(originId);
      result.push(originNode);
      stack.push(originNode);
    }
  }
  return result;
}

// Hash of class types, widget values and mode of every node upstream of this
// Split, read straight from the live graph. Returns null on any failure
// (graph not ready, unexpected shapes, and so on) so callers can fail soft.
export function graphSignature(node) {
  try {
    if (!node?.graph) return null;
    const parts = collectUpstreamNodes(node)
      .map((n) => `${n.id}|${n.type}|${JSON.stringify((n.widgets ?? []).filter(isInputWidget).map((w) => [w.name, w.value]))}|${n.mode}`)
      .sort();
    return hashString(parts.join("\n"));
  } catch {
    return null;
  }
}

// True when a previously stored signature exists and differs from the
// current one. A null/undefined stored or current signature is never stale.
export function isStale(stored, currentSig) {
  if (!stored) return false;
  return (
    currentSig !== null &&
    currentSig !== undefined &&
    stored.sig !== null &&
    stored.sig !== undefined &&
    currentSig !== stored.sig
  );
}

function setPanel(node, text) {
  if (node.kmPanel) node.kmPanel.textContent = text;
}

// Extract "#<id> <class_type>: <message>" lines from a queuePrompt node_errors
// payload (HTTP 400 shape), or null if the shape does not match.
export function nodeErrorLines(nodeErrors) {
  if (!nodeErrors || typeof nodeErrors !== "object") return null;
  const lines = Object.entries(nodeErrors).map(([id, info]) => {
    const classType = info?.class_type ?? "?";
    const message = info?.errors?.[0]?.message ?? "unknown error";
    return `#${id} ${classType}: ${message}`;
  });
  return lines.length ? lines : null;
}

// Fetches and writes the info panel text, guarding against an older response
// arriving after a newer refresh has already started (kmRefreshSeq).
// `deps.fetchPreview({ shape, params })` resolves to `{ text }` or `{ error }`.
// `deps.signature` is the current graph signature (or null), used only for
// the staleness prefix.
export async function refreshPanel(node, { fetchPreview, signature }) {
  const seq = (node.kmRefreshSeq = (node.kmRefreshSeq ?? 0) + 1);
  const stored = node.properties?.km_source; // { shape: [n, h, w, c], sig }
  const stale = isStale(stored, signature);
  const params = Object.fromEntries(PLAN_WIDGETS.map((n) => [n, widget(node, n)?.value]));
  let text;
  try {
    const data = await fetchPreview({ shape: stored?.shape ?? null, params });
    text = data.error ? `Error: ${data.error}` : data.text;
  } catch (e) {
    text = `Preview unavailable: ${e?.message ?? e}`;
  }
  if (stale) text = "Source changed, press Measure (numbers below are from the last measurement)\n" + text;
  if (seq === node.kmRefreshSeq) setPanel(node, text);
  return text;
}

// Wires the preset widget and every PLAN_WIDGETS widget on `node`: selecting
// a preset fills exactly its widgets, editing a widget the preset set flips
// the preset label to "custom", and `onChange(node)` is called for every
// edit (the caller schedules a panel refresh there).
export function wirePresets(node, presets, onChange) {
  let applying = false;
  const presetWidget = widget(node, "preset");
  if (!presetWidget) return;

  chain(presetWidget, "callback", (value) => {
    const values = presets[value];
    if (values) {
      applying = true;
      for (const [name, v] of Object.entries(values)) {
        const w = widget(node, name);
        if (w) w.value = v;
      }
      applying = false;
      node.setDirtyCanvas?.(true, true);
    }
    onChange(node);
  });

  for (const name of PLAN_WIDGETS) {
    const w = widget(node, name);
    if (!w) continue;
    chain(w, "callback", () => {
      // Editing a value the current preset set means the node no longer
      // matches that preset: show "custom" so the label stays honest.
      const setByPreset = presets[presetWidget.value];
      if (!applying && setByPreset && name in setByPreset) {
        presetWidget.value = CUSTOM;
        node.setDirtyCanvas?.(true, true);
      }
      onChange(node);
    });
  }
}

// Resolves a terminal execution event (execution_error / execution_success /
// execution_interrupted) against the node(s) currently in "Measuring..."
// (node.kmMeasurePromptId === detail.prompt_id). Returns null when no node
// in `nodes` matches, otherwise { node, text, clearId, refresh }: `text` (if
// present) should be written to the panel, `clearId` means the caller should
// reset node.kmMeasurePromptId to null, and `refresh` means the caller should
// schedule a panel refresh.
export function handleMeasureEvent(nodes, type, detail) {
  const promptId = detail?.prompt_id;
  if (!promptId) return null;
  const node = (nodes ?? []).find((n) => n.kmMeasurePromptId === promptId);
  if (!node) return null;

  if (type === "execution_error") {
    return {
      node,
      text: `Measure failed in #${detail.node_id} ${detail.node_type}: ${detail.exception_message}`,
      clearId: true,
      refresh: false,
    };
  }
  if (type === "execution_success" || type === "execution_interrupted") {
    return { node, clearId: true, refresh: true };
  }
  return null;
}
