import { test } from "node:test";
import assert from "node:assert/strict";

import {
  CUSTOM,
  chain,
  collectUpstreamNodes,
  executionId,
  graphSignature,
  handleMeasureEvent,
  hashString,
  isInputWidget,
  isStale,
  nodeErrorLines,
  refreshPanel,
  wirePresets,
} from "../../web/js/tiletime_core.js";

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function makeStaticWidget(name, value) {
  return { name, value };
}

// A widget whose setter invokes its own (possibly chained) callback, like a
// real litegraph/DOM widget can.
function makeLiveWidget(name, value) {
  let v = value;
  return {
    name,
    get value() {
      return v;
    },
    set value(nv) {
      v = nv;
      this.callback?.(nv);
    },
  };
}

function makeGraph(nodesById, links) {
  return {
    links,
    getNodeById: (id) => nodesById[id],
  };
}

// ---------------------------------------------------------------------------
// wirePresets
// ---------------------------------------------------------------------------

test("wirePresets: selecting a preset fills exactly its widgets and leaves others alone", () => {
  const presetWidget = makeStaticWidget("preset", "wan");
  const rows = makeLiveWidget("rows", 1);
  const cols = makeLiveWidget("cols", 1);
  const target_width = makeLiveWidget("target_width", 100);
  const node = {
    widgets: [presetWidget, rows, cols, target_width],
    setDirtyCanvas: () => {},
  };
  const presets = { wan: { rows: 2, cols: 3 } };
  const changes = [];
  wirePresets(node, presets, (n) => changes.push(n));

  presetWidget.callback("wan");

  assert.equal(rows.value, 2);
  assert.equal(cols.value, 3);
  assert.equal(target_width.value, 100);
  assert.equal(presetWidget.value, "wan");
  assert.ok(changes.length > 0);
});

test("wirePresets: applying a preset does not switch the preset to custom", () => {
  const presetWidget = makeStaticWidget("preset", "wan");
  const rows = makeLiveWidget("rows", 1);
  const cols = makeLiveWidget("cols", 1);
  const node = { widgets: [presetWidget, rows, cols], setDirtyCanvas: () => {} };
  const presets = { wan: { rows: 2, cols: 3 } };
  wirePresets(node, presets, () => {});

  presetWidget.callback("wan");

  assert.equal(presetWidget.value, "wan");
});

test("wirePresets: editing a preset-set widget switches preset to custom", () => {
  const presetWidget = makeStaticWidget("preset", "wan");
  const rows = makeLiveWidget("rows", 1);
  const cols = makeLiveWidget("cols", 1);
  const node = { widgets: [presetWidget, rows, cols], setDirtyCanvas: () => {} };
  const presets = { wan: { rows: 2, cols: 3 } };
  wirePresets(node, presets, () => {});
  presetWidget.callback("wan");

  rows.value = 5; // simulates the user editing the widget in the UI

  assert.equal(presetWidget.value, CUSTOM);
});

test("wirePresets: editing a widget the preset did not set leaves the preset alone", () => {
  const presetWidget = makeStaticWidget("preset", "wan");
  const rows = makeLiveWidget("rows", 1);
  const cols = makeLiveWidget("cols", 1);
  const target_width = makeLiveWidget("target_width", 100);
  const node = { widgets: [presetWidget, rows, cols, target_width], setDirtyCanvas: () => {} };
  const presets = { wan: { rows: 2, cols: 3 } };
  wirePresets(node, presets, () => {});
  presetWidget.callback("wan");

  target_width.value = 999; // not part of the "wan" preset

  assert.equal(presetWidget.value, "wan");
});

// ---------------------------------------------------------------------------
// isInputWidget
// ---------------------------------------------------------------------------

test("isInputWidget excludes buttons, $$ names and non-serializing widgets", () => {
  assert.equal(isInputWidget({ name: "rows", type: "number", value: 1 }), true);
  assert.equal(isInputWidget({ name: "go", type: "button" }), false);
  assert.equal(isInputWidget({ name: "$$preview", type: "text" }), false);
  assert.equal(isInputWidget({ name: "info", serialize: false }), false);
  assert.equal(isInputWidget({ name: "info2", options: { serialize: false } }), false);
});

// ---------------------------------------------------------------------------
// graphSignature
// ---------------------------------------------------------------------------

function buildBaseGraph() {
  const loader = {
    id: 1,
    type: "LoadVideo",
    mode: 0,
    inputs: [],
    widgets: [makeStaticWidget("file", "a.mp4")],
  };
  const split = {
    id: 2,
    type: "KMTileTimeSplit",
    mode: 0,
    inputs: [{ name: "images", link: 10 }],
    widgets: [],
  };
  const graph = makeGraph({ 1: loader, 2: split }, { 10: { origin_id: 1, origin_slot: 0 } });
  loader.graph = graph;
  split.graph = graph;
  return { graph, loader, split };
}

test("graphSignature: same graph gives same hash", () => {
  const { split } = buildBaseGraph();
  assert.equal(graphSignature(split), graphSignature(split));
});

test("graphSignature: changing an upstream widget value changes it", () => {
  const { split, loader } = buildBaseGraph();
  const before = graphSignature(split);
  loader.widgets[0].value = "b.mp4";
  const after = graphSignature(split);
  assert.notEqual(before, after);
});

test("graphSignature: a transient preview widget does not change it", () => {
  const { split, loader } = buildBaseGraph();
  loader.widgets.push(makeStaticWidget("$$preview", "x"));
  const before = graphSignature(split);
  loader.widgets[loader.widgets.length - 1].value = "y";
  const after = graphSignature(split);
  assert.equal(before, after);

  const { split: split2, loader: loader2 } = buildBaseGraph();
  loader2.widgets.push({ name: "cache", value: "x", serialize: false });
  const before2 = graphSignature(split2);
  loader2.widgets[loader2.widgets.length - 1].value = "z";
  const after2 = graphSignature(split2);
  assert.equal(before2, after2);
});

test("graphSignature: unconnected and downstream nodes do not affect it", () => {
  const { split, graph } = buildBaseGraph();
  const before = graphSignature(split);

  // An unconnected node that nothing references.
  graph.getNodeById2 = () => undefined; // no-op, just documents intent
  // A downstream node whose input is fed BY split (not upstream of it).
  const downstream = { id: 3, type: "SaveVideo", mode: 0, inputs: [{ name: "tiles", link: 20 }], widgets: [] };
  downstream.graph = graph;
  graph.links[20] = { origin_id: 2, origin_slot: 0 };

  const after = graphSignature(split);
  assert.equal(before, after);
});

test("graphSignature: cycles terminate", () => {
  const nodeA = { id: 5, type: "A", mode: 0, inputs: [{ name: "in", link: 50 }], widgets: [] };
  const nodeB = { id: 6, type: "B", mode: 0, inputs: [{ name: "in", link: 51 }], widgets: [] };
  const split = { id: 7, type: "KMTileTimeSplit", mode: 0, inputs: [{ name: "images", link: 52 }], widgets: [] };
  const graph = makeGraph(
    { 5: nodeA, 6: nodeB, 7: split },
    {
      50: { origin_id: 6, origin_slot: 0 }, // A <- B
      51: { origin_id: 5, origin_slot: 0 }, // B <- A (cycle)
      52: { origin_id: 5, origin_slot: 0 }, // split <- A
    }
  );
  nodeA.graph = graph;
  nodeB.graph = graph;
  split.graph = graph;

  const upstream = collectUpstreamNodes(split);
  assert.equal(upstream.length, 2);
  assert.doesNotThrow(() => graphSignature(split));
});

test("graphSignature: errors return null", () => {
  assert.equal(graphSignature(null), null);
  assert.equal(graphSignature({}), null);
  assert.equal(graphSignature({ graph: undefined }), null);
});

test("hashString is deterministic", () => {
  assert.equal(hashString("abc"), hashString("abc"));
  assert.notEqual(hashString("abc"), hashString("abd"));
});

test("chain wraps the original function and calls both", () => {
  const calls = [];
  const obj = { fn: (x) => calls.push(`orig:${x}`) };
  chain(obj, "fn", (x) => calls.push(`extra:${x}`));
  obj.fn("v");
  assert.deepEqual(calls, ["orig:v", "extra:v"]);
});

// ---------------------------------------------------------------------------
// isStale
// ---------------------------------------------------------------------------

test("isStale: null stored or null current is not stale", () => {
  assert.equal(isStale(null, "abc"), false);
  assert.equal(isStale(undefined, "abc"), false);
  assert.equal(isStale({ sig: "abc" }, null), false);
  assert.equal(isStale({ sig: "abc" }, undefined), false);
  assert.equal(isStale({ sig: null }, "abc"), false);
});

test("isStale: different non-null signatures are stale", () => {
  assert.equal(isStale({ sig: "abc" }, "xyz"), true);
  assert.equal(isStale({ sig: "abc" }, "abc"), false);
});

// ---------------------------------------------------------------------------
// executionId
// ---------------------------------------------------------------------------

test("executionId: top-level id", () => {
  const output = { 5: { class_type: "KMTileTimeSplit" } };
  assert.equal(executionId(output, { id: 5 }), "5");
});

test("executionId: unique subgraph id", () => {
  const output = {
    5: { class_type: "SomethingElse" },
    "sg1:5": { class_type: "KMTileTimeSplit" },
  };
  assert.equal(executionId(output, { id: 5 }), "sg1:5");
});

test("executionId: ambiguous match returns null", () => {
  const output = {
    "sg1:5": { class_type: "KMTileTimeSplit" },
    "sg2:5": { class_type: "KMTileTimeSplit" },
  };
  assert.equal(executionId(output, { id: 5 }), null);
});

// ---------------------------------------------------------------------------
// nodeErrorLines
// ---------------------------------------------------------------------------

test("nodeErrorLines: reads node_errors from e.response", () => {
  const e = { response: { node_errors: { 3: { class_type: "Foo", errors: [{ message: "bad shape" }] } } } };
  const lines = nodeErrorLines(e?.response?.node_errors) ?? nodeErrorLines(e?.node_errors);
  assert.deepEqual(lines, ["#3 Foo: bad shape"]);
});

test("nodeErrorLines: reads node_errors from e", () => {
  const e = { node_errors: { 4: { class_type: "Bar", errors: [{ message: "oops" }] } } };
  const lines = nodeErrorLines(e?.response?.node_errors) ?? nodeErrorLines(e?.node_errors);
  assert.deepEqual(lines, ["#4 Bar: oops"]);
});

test("nodeErrorLines: falls back to a generic message and null for no match", () => {
  assert.deepEqual(nodeErrorLines({ 5: { class_type: "Baz", errors: [] } }), ["#5 Baz: unknown error"]);
  assert.equal(nodeErrorLines(undefined), null);
  assert.equal(nodeErrorLines({}), null);
});

// ---------------------------------------------------------------------------
// refreshPanel
// ---------------------------------------------------------------------------

test("refreshPanel: an older response arriving after a newer one does not overwrite the panel", async () => {
  const node = { kmPanel: { textContent: "" }, widgets: [], properties: {} };
  const fetchSlow = () => new Promise((resolve) => setTimeout(() => resolve({ text: "first" }), 30));
  const fetchFast = () => Promise.resolve({ text: "second" });

  const p1 = refreshPanel(node, { fetchPreview: fetchSlow, signature: null });
  const p2 = refreshPanel(node, { fetchPreview: fetchFast, signature: null });

  await p2;
  assert.equal(node.kmPanel.textContent, "second");

  await p1;
  assert.equal(node.kmPanel.textContent, "second");
});

test("refreshPanel: adds the stale prefix when signatures differ", async () => {
  const node = {
    kmPanel: { textContent: "" },
    widgets: [],
    properties: { km_source: { shape: [1, 2, 2, 3], sig: "abc" } },
  };
  await refreshPanel(node, { fetchPreview: async () => ({ text: "plan text" }), signature: "xyz" });
  assert.ok(node.kmPanel.textContent.startsWith("Source changed"));
  assert.ok(node.kmPanel.textContent.endsWith("plan text"));
});

test("refreshPanel: no stale prefix when signatures match", async () => {
  const node = {
    kmPanel: { textContent: "" },
    widgets: [],
    properties: { km_source: { shape: [1, 2, 2, 3], sig: "abc" } },
  };
  await refreshPanel(node, { fetchPreview: async () => ({ text: "plan text" }), signature: "abc" });
  assert.equal(node.kmPanel.textContent, "plan text");
});

test("refreshPanel: fetch error shows Preview unavailable", async () => {
  const node = { kmPanel: { textContent: "" }, widgets: [], properties: {} };
  await refreshPanel(node, {
    fetchPreview: async () => {
      throw new Error("boom");
    },
    signature: null,
  });
  assert.ok(node.kmPanel.textContent.startsWith("Preview unavailable"));
});

// ---------------------------------------------------------------------------
// handleMeasureEvent
// ---------------------------------------------------------------------------

test("handleMeasureEvent: execution_error with matching prompt_id sets failure text and clears the id", () => {
  const node = { kmMeasurePromptId: "p1" };
  const result = handleMeasureEvent([node], "execution_error", {
    prompt_id: "p1",
    node_id: 9,
    node_type: "Foo",
    exception_message: "bad",
  });
  assert.equal(result.node, node);
  assert.equal(result.text, "Measure failed in #9 Foo: bad");
  assert.equal(result.clearId, true);
  assert.equal(result.refresh, false);
});

test("handleMeasureEvent: execution_success clears the id and requests a refresh", () => {
  const node = { kmMeasurePromptId: "p2" };
  const result = handleMeasureEvent([node], "execution_success", { prompt_id: "p2" });
  assert.equal(result.node, node);
  assert.equal(result.clearId, true);
  assert.equal(result.refresh, true);
  assert.equal(result.text, undefined);
});

test("handleMeasureEvent: execution_interrupted clears the id and requests a refresh", () => {
  const node = { kmMeasurePromptId: "p2b" };
  const result = handleMeasureEvent([node], "execution_interrupted", { prompt_id: "p2b" });
  assert.equal(result.clearId, true);
  assert.equal(result.refresh, true);
});

test("handleMeasureEvent: non-matching prompt_id does nothing", () => {
  const node = { kmMeasurePromptId: "p3" };
  const result = handleMeasureEvent([node], "execution_success", { prompt_id: "other" });
  assert.equal(result, null);
});
