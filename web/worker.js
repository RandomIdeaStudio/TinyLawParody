// Inference off the main thread. Loads web/model/*, then streams generated text.
/* global importScripts, TinyTokenizer, TinyGPTLib, TinyPrompt */
"use strict";
importScripts("tokenizer.js", "transformer.js", "prompt.js");

let model = null;
let tokenizer = null;
let config = null;
let stopRequested = false;
let busy = false;

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url);
  if (!res.ok) throw new Error("HTTP " + res.status + " for " + url);
  const total = Number(res.headers.get("Content-Length")) || 0;
  if (!res.body || !res.body.getReader) return res.arrayBuffer();
  const reader = res.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    onProgress(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.length;
  }
  return out.buffer;
}

async function load() {
  config = await (await fetch("model/config.json")).json();
  const tokData = await (await fetch("model/tokenizer.json")).json();
  tokenizer = new TinyTokenizer.BPETokenizer(tokData);
  const buf = await fetchWithProgress("model/model.bin", (loaded, total) =>
    postMessage({ type: "progress", loaded, total })
  );
  model = new TinyGPTLib.TinyGPT(config, buf);
  postMessage({ type: "ready", version: config.version, sampling: config.sampling });
}

function yieldToEvents() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

async function generate(msg) {
  busy = true;
  stopRequested = false;
  const s = config.sampling;
  const prompt = TinyPrompt.buildPrompt(config.prompt, msg.title, msg.promptOptions);
  postMessage({ type: "prompt", text: prompt });
  const promptIds = tokenizer.encode(prompt);
  const ctx = config.model.context;
  const keep = Math.floor(ctx / 2);
  let all = promptIds.slice(-ctx);
  const generated = [];
  const random = TinyGPTLib.rng(msg.seed);
  let logits = model.prefill(all);
  let reason = "length";
  let lastText = "";
  for (let i = 0; i < (msg.maxNewTokens || s.max_new_tokens); i++) {
    if (stopRequested) {
      reason = "stopped";
      break;
    }
    const next = TinyGPTLib.sample(logits, msg.temperature, s.top_k, random);
    if (next === tokenizer.eotId) {
      reason = "end";
      break;
    }
    generated.push(next);
    all.push(next);
    const text = tokenizer.decode(generated);
    // hold back an incomplete multi-byte character
    if (!text.endsWith("�") && text !== lastText) {
      lastText = text;
      postMessage({ type: "text", text });
    }
    if (model.pos >= ctx) {
      all = all.slice(-keep); // slide the window: recompute the cache from recent tokens
      logits = model.prefill(all);
    } else {
      logits = model.step(next);
    }
    await yieldToEvents();
  }
  postMessage({ type: "text", text: tokenizer.decode(generated) });
  postMessage({ type: "done", reason });
  busy = false;
}

onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === "load") await load();
    else if (msg.type === "stop") stopRequested = true;
    else if (msg.type === "generate" && !busy) await generate(msg);
  } catch (err) {
    busy = false;
    postMessage({ type: "error", message: String(err && err.message ? err.message : err) });
  }
};
