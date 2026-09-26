// KM Tile & Time Split and Merge: ComfyUI glue (extension registration, app/api wiring).
import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import {
  chain,
  executionId,
  graphSignature,
  handleMeasureEvent,
  migrateLegacyWidgetValues,
  nodeErrorLines,
  refreshPanel,
  syncOverlayWidget,
  wirePresets,
} from "./tiletime_core.js";

const NODE_TYPE = "KMTileTimeSplit";
const MERGE_TYPE = "KMTileTimeMerge";
const REFRESH_DELAY_MS = 150;

function setPanel(node, text) {
  if (node.kmPanel) node.kmPanel.textContent = text;
}

async function fetchPreview({ shape, params }) {
  const res = await api.fetchApi("/km_tiletime/preview", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ shape, params }),
  });
  return res.json();
}

async function refresh(node) {
  await refreshPanel(node, { fetchPreview, signature: graphSignature(node) });
}

function scheduleRefresh(node) {
  clearTimeout(node.kmRefreshTimer);
  node.kmRefreshTimer = setTimeout(() => refresh(node), REFRESH_DELAY_MS);
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

// Merge: overlay_on mirrors whether the overlay output is linked. ComfyUI caches
// Merge by its inputs, so this flag is what makes a newly connected overlay run
// again instead of returning the cached placeholder. The widget is hidden.
function hideWidget(node, name) {
  const w = node.widgets?.find((x) => x.name === name);
  if (!w) return;
  w.hidden = true;
  w.computeSize = () => [0, -4];
}

function syncOverlay(node) {
  try {
    if (syncOverlayWidget(node)) node.setDirtyCanvas?.(true, true);
  } catch {
    // fail soft
  }
}

function registerMerge(nodeType) {
  chain(nodeType.prototype, "onNodeCreated", function () {
    try {
      hideWidget(this, "overlay_on");
    } catch {
      // fail soft
    }
    syncOverlay(this);
  });

  chain(nodeType.prototype, "onConfigure", function () {
    syncOverlay(this);
  });

  chain(nodeType.prototype, "onConnectionsChange", function () {
    syncOverlay(this);
  });
}

app.registerExtension({
  name: "km.tiletime.split",

  setup() {
    // Upstream edits (a new file in the loader, rewiring) do not touch Split's
    // widgets, so also refresh on every graph change.
    api.addEventListener("graphChanged", () => {
      for (const node of app.graph?.findNodesByType?.(NODE_TYPE) ?? []) scheduleRefresh(node);
    });

    // Never let the panel stay on "Measuring..." forever: clear the pending
    // id and refresh on any terminal event for that prompt.
    const onMeasureEvent = (type) => (e) => {
      try {
        const nodes = app.graph?.findNodesByType?.(NODE_TYPE) ?? [];
        const result = handleMeasureEvent(nodes, type, e?.detail ?? {});
        if (!result) return;
        if (result.text !== undefined) setPanel(result.node, result.text);
        if (result.clearId) result.node.kmMeasurePromptId = null;
        if (result.refresh) scheduleRefresh(result.node);
      } catch {
        // fail soft
      }
    };

    api.addEventListener("execution_error", onMeasureEvent("execution_error"));
    api.addEventListener("execution_success", onMeasureEvent("execution_success"));
    api.addEventListener("execution_interrupted", onMeasureEvent("execution_interrupted"));
  },

  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData.name === MERGE_TYPE) {
      registerMerge(nodeType);
      return;
    }
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
      wirePresets(this, presets, scheduleRefresh);
      const min = this.computeSize();
      this.setSize([Math.max(this.size[0], 380), Math.max(this.size[1], min[1])]);
      scheduleRefresh(this);
    });

    // Saved workflows restore properties (the measured source) in configure.
    chain(nodeType.prototype, "onConfigure", function () {
      if (migrateLegacyWidgetValues(this)) this.setDirtyCanvas?.(true, true);
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
