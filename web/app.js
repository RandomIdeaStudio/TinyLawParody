// Page logic: language, controls, About dialog; inference runs in worker.js.
"use strict";

const LANGS = ["cs", "en"];
const STORAGE_KEY = "tinylaw.lang";
let strings = {};
let lang = "en";
let worker = null;
let ready = false;
let generating = false;
let lastTitle = "";
let version = "";

const $ = (id) => document.getElementById(id);

function t(key, vars) {
  let s = Object.prototype.hasOwnProperty.call(strings, key) ? strings[key] : key;
  if (vars) s = s.replace(/\{(\w+)\}/g, (m, k) => (k in vars ? String(vars[k]) : m));
  return s;
}

function storedLanguage() {
  try {
    const v = localStorage.getItem(STORAGE_KEY);
    return LANGS.includes(v) ? v : null;
  } catch (e) {
    return null;
  }
}

function storeLanguage(v) {
  try {
    localStorage.setItem(STORAGE_KEY, v);
  } catch (e) {
    /* storage unavailable: the choice lasts for this visit only */
  }
}

function browserLanguage() {
  const list = navigator.languages && navigator.languages.length ? navigator.languages : [navigator.language];
  for (const l of list) {
    const base = String(l || "").toLowerCase().split("-")[0];
    if (LANGS.includes(base)) return base;
  }
  return "en";
}

async function setLanguage(next) {
  const res = await fetch("locales/" + next + ".json");
  strings = await res.json();
  lang = next;
  document.documentElement.lang = next;
  $("lang").value = next;
  for (const el of document.querySelectorAll("[data-i18n]")) el.textContent = t(el.dataset.i18n);
  for (const el of document.querySelectorAll("[data-i18n-placeholder]")) {
    el.placeholder = t(el.dataset.i18nPlaceholder);
  }
  for (const el of document.querySelectorAll("[data-i18n-aria-label]")) {
    el.setAttribute("aria-label", t(el.dataset.i18nAriaLabel));
  }
  $("version").textContent = version || "–";
  updateCraziness();
  if (!ready) setStatus(t("status.loading"));
  else if (!generating) setStatus(t("status.ready"));
}

function setStatus(text) {
  $("status").textContent = text;
}

function updateCraziness() {
  const v = Number($("craziness").value);
  const word = v < 0.8 ? t("form.craziness_calm") : v <= 1.1 ? t("form.craziness_normal") : t("form.craziness_wild");
  const text = v.toFixed(2) + " – " + word;
  $("craziness-value").textContent = text;
  $("craziness").setAttribute("aria-valuetext", text);
}

function setGenerating(on) {
  generating = on;
  $("generate").disabled = on || !ready;
  $("again").disabled = on || !ready || !lastTitle;
  $("stop").disabled = !on;
  $("output").setAttribute("aria-busy", on ? "true" : "false");
}

function randomInt(lo, hi) {
  return lo + Math.floor(Math.random() * (hi - lo + 1));
}

function start(title) {
  if (!ready || generating) return;
  lastTitle = title;
  const now = new Date();
  const out = $("output");
  out.textContent = "";
  setGenerating(true);
  setStatus(t("status.generating"));
  worker.postMessage({
    type: "generate",
    title,
    temperature: Number($("craziness").value),
    seed: randomInt(1, 2147483646),
    promptOptions: {
      number: randomInt(1, 499),
      year: now.getFullYear(),
      day: now.getDate(),
      month: now.getMonth() + 1,
      part: randomInt(1, 199),
    },
  });
}

let promptNode = null;
let textNode = null;

function onWorkerMessage(e) {
  const msg = e.data;
  if (msg.type === "progress") {
    const pct = msg.total ? Math.round((100 * msg.loaded) / msg.total) : null;
    setStatus(pct === null ? t("status.loading") : t("status.loading_progress", { percent: pct }));
  } else if (msg.type === "ready") {
    ready = true;
    version = msg.version;
    $("version").textContent = version;
    setGenerating(false);
    setStatus(t("status.ready"));
  } else if (msg.type === "prompt") {
    const out = $("output");
    out.textContent = "";
    promptNode = document.createElement("span");
    promptNode.className = "prompt-part";
    promptNode.textContent = msg.text;
    textNode = document.createElement("span");
    out.append(promptNode, textNode);
  } else if (msg.type === "text") {
    if (textNode) textNode.textContent = msg.text;
  } else if (msg.type === "done") {
    setGenerating(false);
    setStatus(msg.reason === "stopped" ? t("status.stopped") : t("status.done"));
  } else if (msg.type === "error") {
    console.error("worker error:", msg.message);
    setGenerating(false);
    setStatus(ready ? t("status.error") : t("status.load_error"));
  }
}

async function init() {
  await setLanguage(storedLanguage() || browserLanguage());
  setGenerating(false);
  $("lang").addEventListener("change", (e) => {
    storeLanguage(e.target.value);
    setLanguage(e.target.value);
  });
  $("craziness").addEventListener("input", updateCraziness);
  $("form").addEventListener("submit", (e) => {
    e.preventDefault();
    const title = $("title").value.trim();
    if (title) start(title);
  });
  $("again").addEventListener("click", () => lastTitle && start(lastTitle));
  $("stop").addEventListener("click", () => worker && worker.postMessage({ type: "stop" }));
  for (const chip of document.querySelectorAll("[data-example]")) {
    chip.addEventListener("click", () => {
      $("title").value = t(chip.dataset.example);
      $("title").focus();
    });
  }
  $("about-open").addEventListener("click", () => $("about").showModal());
  worker = new Worker("worker.js");
  worker.onmessage = onWorkerMessage;
  worker.onerror = (e) => {
    console.error(e);
    setStatus(t("status.load_error"));
  };
  worker.postMessage({ type: "load" });
}

init();
