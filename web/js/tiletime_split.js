// KM Tile & Time Split: preset filling, info panel and Measure source.
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_TYPE = "KMTileTimeSplit";
const CUSTOM = "custom";
// Widgets whose values feed the layout (sent to the preview route).
const PLAN_WIDGETS = [
  "tile_mode", "rows", "cols", "target_width", "target_height", "overlap_mode",
  "overlap", "multiple_of", "chunk_frames", "chunk_overlap", "frame_rule",
];
const REFRESH_DELAY_MS = 150;

function chain(obj, name, fn) {
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

// Small stable string hash (djb2) for the upstream signature.
function hashString(s) {
  let h = 5381;
  for (let i = 0; i < s.length; i++) h = ((h << 5) + h + s.charCodeAt(i)) | 0;
  return (h >>> 0).toString(16);
}

// Id of this node in the API prompt. Nodes inside subgraphs get "parent:child" ids.
function executionId(output, node) {
  const id = String(node.id);
  if (output[id]?.class_type === NODE_TYPE) return id;
  const hits = Object.keys(output).filter(
    (k) => k.endsWith(":" + id) && output[k].class_type === NODE_TYPE
  );
  return hits.length === 1 ? hits[0] : null;
}

// Walk input links upstream of `node` inside its own graph, collecting every
// node reached (no dependency on the API-format prompt).
function collectUpstreamNodes(node) {
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
function graphSignature(node) {
  try {
    if (!node?.graph) return null;
    const parts = collectUpstreamNodes(node)
      .map((n) => `${n.id}|${n.type}|${JSON.stringify((n.widgets ?? []).map((w) => w.value))}|${n.mode}`)
      .sort();
    return hashString(parts.join("\n"));
  } catch {
    return null;
  }
}

function setPanel(node, text) {
  if (node.kmPanel) node.kmPanel.textContent = text;
}

async function refresh(node) {
  const seq = (node.kmRefreshSeq = (node.kmRefreshSeq ?? 0) + 1);
  const stored = node.properties?.km_source; // { shape: [n, h, w, c], sig }
  let stale = false;
  if (stored) {
    const sig = graphSignature(node);
    stale = sig !== null && stored.sig !== null && stored.sig !== undefined && sig !== stored.sig;
  }
  const params = Object.fromEntries(PLAN_WIDGETS.map((n) => [n, widget(node, n)?.value]));
  let text;
  try {
    const res = await api.fetchApi("/km_tiletime/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ shape: stored?.shape ?? null, params }),
    });
    const data = await res.json();
    text = data.error ? `Error: ${data.error}` : data.text;
  } catch (e) {
    text = `Preview unavailable: ${e?.message ?? e}`;
  }
  if (stale) text = "Source changed, press Measure (numbers below are from the last measurement)\n" + text;
  if (seq === node.kmRefreshSeq) setPanel(node, text);
}

function scheduleRefresh(node) {
  clearTimeout(node.kmRefreshTimer);
  node.kmRefreshTimer = setTimeout(() => refresh(node), REFRESH_DELAY_MS);
}

// Extract "#<id> <class_type>: <message>" lines from a queuePrompt node_errors
// payload (HTTP 400 shape), or null if the shape does not match.
function nodeErrorLines(nodeErrors) {
  if (!nodeErrors || typeof nodeErrors !== "object") return null;
  const lines = Object.entries(nodeErrors).map(([id, info]) => {
    const classType = info?.class_type ?? "?";
    const message = info?.errors?.[0]?.message ?? "unknown error";
    return `#${id} ${classType}: ${message}`;
  });
  return lines.length ? lines : null;
}

async function measure(node) {
  try {
    const caps = await (await api.fetchApi("/km_tiletime/caps")).json();
    if (!caps.partial) {
      setPanel(node, "This ComfyUI version cannot run part of a workflow.\nRun the workflow once and the panel fills in.");
      return;
    }
    const prompt = await app.graphToPrompt();
    const id = executionId(prompt.output, node);
    if (!id) {
      setPanel(node, "Could not find this node in the workflow (is it bypassed or muted?)");
      return;
    }
    setPanel(node, "Measuring: running only the nodes that feed this Split...");
    const result = await api.queuePrompt(0, prompt, { partialExecutionTargets: [id] });
    node.kmMeasurePromptId = result?.prompt_id ?? null;
  } catch (e) {
    const lines = nodeErrorLines(e?.response?.node_errors) ?? nodeErrorLines(e?.node_errors);
    setPanel(node, lines ? lines.join("\n") : `Measure failed: ${e?.message ?? e}`);
    node.kmMeasurePromptId = null;
  }
}

function wirePresets(node, presets) {
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
    scheduleRefresh(node);
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
      scheduleRefresh(node);
    });
  }
}

app.registerExtension({
  name: "km.tiletime.split",

  setup() {
    // Upstream edits (a new file in the loader, rewiring) do not touch Split's
    // widgets, so also refresh on every graph change.
    api.addEventListener("graphChanged", () => {
      for (const node of app.graph?.findNodesByType?.(NODE_TYPE) ?? []) scheduleRefresh(node);
    });

    const findMeasuringNode = (promptId) => {
      if (!promptId) return null;
      for (const node of app.graph?.findNodesByType?.(NODE_TYPE) ?? []) {
        if (node.kmMeasurePromptId === promptId) return node;
      }
      return null;
    };

    // Never let the panel stay on "Measuring..." forever: clear the pending
    // id and refresh on any terminal event for that prompt.
    api.addEventListener("execution_error", (e) => {
      try {
        const detail = e?.detail ?? {};
        const node = findMeasuringNode(detail.prompt_id);
        if (!node) return;
        setPanel(node, `Measure failed in #${detail.node_id} ${detail.node_type}: ${detail.exception_message}`);
        node.kmMeasurePromptId = null;
      } catch {
        // fail soft
      }
    });

    api.addEventListener("execution_success", (e) => {
      try {
        const node = findMeasuringNode(e?.detail?.prompt_id);
        if (!node) return;
        node.kmMeasurePromptId = null;
        scheduleRefresh(node);
      } catch {
        // fail soft
      }
    });

    api.addEventListener("execution_interrupted", (e) => {
      try {
        const node = findMeasuringNode(e?.detail?.prompt_id);
        if (!node) return;
        node.kmMeasurePromptId = null;
        scheduleRefresh(node);
      } catch {
        // fail soft
      }
    });
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name !== NODE_TYPE) return;
    const spec = nodeData.input?.required?.preset;
    const presets = (Array.isArray(spec) && spec[1]?.km_presets) || {};

    chain(nodeType.prototype, "onNodeCreated", function () {
      const panel = document.createElement("div");
      Object.assign(panel.style, {
        font: "11px monospace",
        whiteSpace: "pre-wrap",
        overflow: "auto",
        padding: "6px",
        boxSizing: "border-box",
        color: "var(--input-text, #ddd)",
        background: "var(--comfy-input-bg, #222)",
        borderRadius: "4px",
      });
      this.kmPanel = panel;
      const measureButton = this.addWidget("button", "Measure source", null, () => measure(this));
      measureButton.serialize = false;
      this.addDOMWidget("km_info", "KM_INFO", panel, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 110,
      });
      wirePresets(this, presets);
      const min = this.computeSize();
      this.setSize([Math.max(this.size[0], 380), Math.max(this.size[1], min[1])]);
      scheduleRefresh(this);
    });

    // Saved workflows restore properties (the measured source) in configure.
    chain(nodeType.prototype, "onConfigure", function () {
      scheduleRefresh(this);
    });

    chain(nodeType.prototype, "onConnectionsChange", function () {
      scheduleRefresh(this);
    });

    chain(nodeType.prototype, "onExecuted", function (message) {
      const shape = message?.km_source?.[0];
      if (!shape) return;
      const sig = graphSignature(this);
      this.properties = this.properties ?? {};
      this.properties.km_source = { shape, sig };
      scheduleRefresh(this);
    });
  },
});
