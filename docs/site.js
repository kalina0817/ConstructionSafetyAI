"use strict";
const el = id => document.getElementById(id);
const methods = ["Scene-graph rules", "MiniCPM-V", "Qwen3-VL"];
const outcomes = {POTENTIAL:"Possible safety concern",HIGH_CONFIDENCE_ALERT:"Strong evidence of a possible concern",NO_SUPPORTED:"No supported concern found",INSUFFICIENT:"Too little evidence"};
const gearNames = {helmet:"Hard hat",high_visibility_vest:"Safety vest",gloves:"Gloves",boots:"Boots",eye_protection:"Eye protection"};
const states = {present:"Visible",potentially_missing:"May be missing",unassessable:"Cannot tell"};
let photos, reportURL;
function node(tag, text) {
  const item = document.createElement(tag);
  item.textContent = text;
  return item;
}
function detail(parent, title) {
  const card = document.createElement("details");
  card.append(node("summary", title));
  parent.append(card);
  return card;
}
function descriptions(card, fields) {
  const list = document.createElement("dl");
  for (const [label, value] of fields) {
    if (value) list.append(node("dt", label), node("dd", value));
  }
  card.append(list);
}
function setImage() {
  const id = el("photo").value;
  const marked = el("boxes").checked;
  el("scene").src = "photos/" + id + (marked ? "-marked.jpg" : ".jpg");
  el("scene").alt = "Construction site in " + id.replace("_", " ") + (marked ? " with detected objects marked" : "");
  el("caption").textContent = id.replace("_", " ");
}
function render() {
  const id = el("photo").value;
  const method = el("method").value;
  const photo = photos[id];
  const run = photo.methods[method];
  setImage();
  el("selection").textContent = "YOLO11n · " + method;
  el("status").textContent = id.replace("_", " ") + " · Saved result from " + method;
  el("outcome").textContent = outcomes[run.outcome.outcome] || "Outcome unavailable";
  el("explanation").textContent = run.outcome.explanation;
  el("counts").textContent = photo.objects.length + " object labels · " + run.findings.length + " possible concerns";
  const findings = el("findings");
  findings.replaceChildren();
  if (!run.findings.length) findings.append(node("p","This method raised no alerts for this photo."));
  for (const f of run.findings) {
    const card = detail(findings, f.finding_category.replaceAll("_"," "));
    const ids = (f.source_object_ids || []).join(", ");
    descriptions(card,[["Observation",f.observation],["Evidence",f.evidence?.visual_evidence || ids],["Uncertainty",f.limitation],["Suggested action",f.recommendation]]);
  }
  const gear = el("gear");
  gear.replaceChildren();
  const workers = new Map();
  for (const row of run.gear) {
    if (!workers.has(row.worker)) workers.set(row.worker, detail(gear,"Worker · " + row.worker));
    descriptions(workers.get(row.worker), [[(gearNames[row.item] || row.item) + " · " + (states[row.status] || row.status),row.visual_basis_or_uncertainty]]);
  }
  if (!workers.size) gear.append(node("p","No worker was identified for a protective gear check."));
  if (reportURL) URL.revokeObjectURL(reportURL);
  const result = {photo:id,detection_model:"YOLO11n",reasoning_model:method,objects:photo.objects,...run};
  reportURL = URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:"application/json"}));
  el("download").href = reportURL;
  el("download").download = id + "-" + method.replaceAll(" ","-") + ".json";
}
async function start() {
  const response = await fetch("results.json");
  if (!response.ok) throw new Error("Results unavailable");
  photos = await response.json();
  for (const id of Object.keys(photos)) el("photo").add(new Option(id.replace("_"," "),id));
  for (const method of methods) el("method").add(new Option(method,method));
  for (const id of ["photo","method"]) {
    el(id).disabled = false;
    el(id).addEventListener("change",render);
  }
  el("boxes").addEventListener("change",setImage);
  render();
  el("workspace").hidden = false;
}
start().catch(()=>{el("status").textContent="Could not load the saved checks. Refresh the page to try again.";});
