// Page logic: language, the chat with the clerk, settings, About dialog.
// Inference runs in worker.js; chat.js turns messages into model requests and replies.
"use strict";

const LANGS = ["cs", "en"];
const STORAGE_KEY = "tinylaw.lang";
let strings = {};
let lang = "en";
let worker = null;
let ready = false;
let generating = false;
let version = "";

// Conversation shown on the page. The clerk does not remember it: each message is handled alone.
// { role: "user" | "clerk", text, number } ; number = the § shown on a clerk reply.
const conversation = [];
let current = null; // the clerk reply being generated: { entry, request, node }
let lastUserMessage = "";

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
  renderConversation();
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
  $("send").disabled = on || !ready;
  $("again").disabled = on || !ready || !lastClerkEntry();
  $("stop").disabled = !on;
  $("clear").disabled = on;
  $("messages").setAttribute("aria-busy", on ? "true" : "false");
}

function randomInt(lo, hi) {
  return lo + Math.floor(Math.random() * (hi - lo + 1));
}

function lastClerkEntry() {
  const last = conversation[conversation.length - 1];
  return last && last.role === "clerk" && !last.pending ? last : null;
}

function nextParagraphNumber() {
  return conversation.filter((m) => m.role === "clerk").length + 1;
}

// --- rendering ---------------------------------------------------------------

function messageNode(entry) {
  const li = document.createElement("li");
  li.className = "msg " + entry.role;
  const who = document.createElement("span");
  who.className = "who";
  who.textContent = entry.role === "user" ? t("chat.you") : t("chat.clerk");
  const body = document.createElement("p");
  body.className = "text";
  if (entry.role === "clerk" && entry.number) {
    const para = document.createElement("span");
    para.className = "para";
    para.textContent = "§ " + entry.number + " ";
    body.append(para);
  }
  const text = document.createElement("span");
  text.className = "body";
  text.textContent = entry.pending ? t("chat.typing") : entry.text;
  body.append(text);
  li.append(who, body);
  return li;
}

function renderConversation() {
  const list = $("messages");
  list.textContent = "";
  list.append(messageNode({ role: "clerk", text: t("chat.welcome") }));
  for (const entry of conversation) {
    const node = messageNode(entry);
    if (entry === (current && current.entry)) current.node = node;
    list.append(node);
  }
  scrollToEnd();
}

// Scroll to the bottom of the page: the newest message then sits just above the composer
// (scrolling the message itself would hide it behind the sticky composer).
function scrollToEnd() {
  $("status").scrollIntoView({ block: "end", behavior: "auto" });
}

function updateCurrent(text) {
  if (!current) return;
  current.entry.text = text;
  if (current.node) current.node.querySelector(".body").textContent = text;
  scrollToEnd();
}

// --- conversation ------------------------------------------------------------

function ask(rawMessage, { replaceLast = false } = {}) {
  if (!ready || generating) return;
  const message = String(rawMessage).trim();
  if (!message) return;
  if (replaceLast) {
    conversation.pop(); // drop the previous clerk reply to this message
  } else {
    conversation.push({ role: "user", text: message });
  }
  lastUserMessage = message;
  const request = TinyChat.chatRequest(message);
  const entry = { role: "clerk", text: "", number: nextParagraphNumber(), pending: true };
  conversation.push(entry);
  current = { entry, request, node: null };
  renderConversation();
  setGenerating(true);
  setStatus(t("status.generating"));
  const now = new Date();
  worker.postMessage({
    type: "generate",
    title: request.title,
    promptSuffix: request.promptSuffix,
    stopAt: request.stopAt,
    maxNewTokens: request.maxNewTokens,
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

function finishCurrent(generated, reason) {
  if (!current) return;
  current.entry.pending = false;
  const text = TinyChat.formatReply(current.request.promptSuffix, generated, reason);
  updateCurrent(text);
  current = null;
}

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
  } else if (msg.type === "text") {
    if (current) {
      current.entry.pending = false;
      current.generated = msg.text;
      updateCurrent(TinyChat.previewReply(current.request.promptSuffix, msg.text));
    }
  } else if (msg.type === "done") {
    finishCurrent(current ? current.generated || "" : "", msg.reason);
    setGenerating(false);
    setStatus(msg.reason === "stopped" ? t("status.stopped") : t("status.replied"));
  } else if (msg.type === "error") {
    console.error("worker error:", msg.message);
    if (current) {
      conversation.pop();
      current = null;
      renderConversation();
    }
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
  $("composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const message = $("message").value.trim();
    if (!message || !ready || generating) return;
    $("message").value = "";
    ask(message);
  });
  $("message").addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      $("composer").requestSubmit();
    }
  });
  $("again").addEventListener("click", () => lastClerkEntry() && ask(lastUserMessage, { replaceLast: true }));
  $("stop").addEventListener("click", () => worker && worker.postMessage({ type: "stop" }));
  $("clear").addEventListener("click", () => {
    if (generating) return;
    conversation.length = 0;
    lastUserMessage = "";
    renderConversation();
    setGenerating(false);
    $("message").focus();
  });
  for (const chip of document.querySelectorAll("[data-example]")) {
    chip.addEventListener("click", () => {
      $("message").value = t(chip.dataset.example);
      $("message").focus();
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
