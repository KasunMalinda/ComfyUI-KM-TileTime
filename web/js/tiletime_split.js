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

// Hash of class types and inputs of every node upstream of this Split.
function upstreamSignature(output, execId) {
  const seen = new Set();
  const stack = [execId];
  while (stack.length) {
    const inputs = output[stack.pop()]?.inputs ?? {};
    for (const v of Object.values(inputs)) {
      if (Array.isArray(v) && v.length === 2 && output[String(v[0])] && !seen.has(String(v[0]))) {
        seen.add(String(v[0]));
        stack.push(String(v[0]));
      }
    }
  }
  const parts = [...seen].sort().map(
    (id) => `${id}|${output[id].class_type}|${JSON.stringify(output[id].inputs)}`
  );
  return hashString(parts.join("\n"));
}

async function currentSignature(node) {
  const { output } = await app.graphToPrompt();
  const id = executionId(output, node);
  return id ? upstreamSignature(output, id) : null;
}

function setPanel(node, text) {
  if (node.kmPanel) node.kmPanel.textContent = text;
}

async function refresh(node) {
  const stored = node.properties?.km_source; // { shape: [n, h, w, c], sig }
  let stale = false;
  if (stored) {
    try {
      const sig = await currentSignature(node);
      stale = sig !== null && sig !== stored.sig;
    } catch {
      stale = false; // graph not ready yet; show the stored numbers
    }
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
  setPanel(node, text);
}

function scheduleRefresh(node) {
  clearTimeout(node.kmRefreshTimer);
  node.kmRefreshTimer = setTimeout(() => refresh(node), REFRESH_DELAY_MS);
}

async function measure(node) {
  setPanel(node, "Measuring: running only the nodes that feed this Split...");
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
    await api.queuePrompt(0, prompt, { partialExecutionTargets: [id] });
  } catch (e) {
    setPanel(node, `Measure failed: ${e?.message ?? e}`);
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
      this.setSize([Math.max(this.size[0], 380), this.size[1]]);
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
      currentSignature(this)
        .catch(() => null)
        .then((sig) => {
          this.properties = this.properties ?? {};
          this.properties.km_source = { shape, sig };
          scheduleRefresh(this);
        });
    });
  },
});
