// Chat with the clerk: turns a visitor's message into a model request and the model's
// continuation into a short reply. Works in the page and in Node (tests).
//
// The model only knows Czech laws, so every message is wrapped as a law "on handling the
// submission <message>" and the reply is started for the model with a fixed official phrase
// ("Podání „…“ se zamítá, neboť"), which it then continues. These Czech strings are model
// input in the model's own language, not page text, so they are not in the locale files.
(function (root) {
  "use strict";

  const TITLE = "o vyřízení podání „{message}“";
  const REPLY_START = "(1)\n\nPodání „{message}“ {opening}";
  // No trailing space: the tokenizer joins a space with the next word, so a prompt that ends
  // with a space makes the model produce broken words.
  const OPENINGS = [
    "se zamítá, neboť",
    "je přípustné, jen jestliže",
    "se vyřizuje do 30 dnů, a to",
    "se postoupí příslušnému úřadu, který",
    "se považuje za",
    "se vrací žadateli, protože",
    "nemá odkladný účinek, neboť",
    "se vyhovuje, pokud",
  ];
  const STOP_AT = ["\n§ 2", "\n\n(2)", "\n\n§", "\nČl. ", "\nČÁST "];
  const MAX_MESSAGE = 120;
  const MAX_NEW_TOKENS = 80;

  function sanitizeMessage(raw) {
    let s = String(raw)
      .replace(/[„“”"«»]/gu, "")
      .replace(/\s+/gu, " ")
      .trim();
    if (s.length > MAX_MESSAGE) {
      s = s.slice(0, MAX_MESSAGE);
      const space = s.lastIndexOf(" ");
      if (space > MAX_MESSAGE / 2) s = s.slice(0, space);
      s += "…";
    }
    return s;
  }

  function fill(template, values) {
    return template.replace(/\{(\w+)\}/g, (m, k) => (k in values ? String(values[k]) : m));
  }

  // random: a function returning a number in [0, 1), used to pick the opening.
  function chatRequest(rawMessage, random) {
    const message = sanitizeMessage(rawMessage);
    const opening = OPENINGS[Math.floor((random || Math.random)() * OPENINGS.length) % OPENINGS.length];
    return {
      message,
      title: fill(TITLE, { message }),
      promptSuffix: fill(REPLY_START, { message, opening }),
      stopAt: STOP_AT.slice(),
      maxNewTokens: MAX_NEW_TOKENS,
    };
  }

  // Joins the started reply and the model's continuation into one or two sentences.
  // reason: "end" (stopped at a paragraph boundary or end of text), "length", "stopped".
  // Czech legal abbreviations whose dot does not end a sentence ("odst. 1", "č. 12/2026 Sb.").
  const ABBREVIATIONS = new Set(["odst", "písm", "č", "čl", "sb", "zák", "tzv", "např", "resp", "apod", "mj", "popř", "příp", "kč", "r", "s", "v", "ust", "hl", "ods"]);

  function isAbbreviation(before) {
    const word = (before.match(/([\p{L}\p{N}]+)$/u) || [])[1];
    if (!word) return false;
    if (/^\p{N}+$/u.test(word)) return true; // "1." as an ordinal or list number
    return ABBREVIATIONS.has(word.toLowerCase());
  }

  function normalize(s) {
    return s.replace(/\s+/gu, " ");
  }

  function replyHead(promptSuffix) {
    return normalize(promptSuffix.replace(/^\(1\)\s*/u, "")).trim();
  }

  // What the bubble shows while the model is still writing.
  function previewReply(promptSuffix, generated) {
    return normalize(replyHead(promptSuffix) + generated).trim();
  }

  // The model sometimes repeats the end of the quoted message ("?“"); drop closing quotes
  // that have no opening quote in the model's part of the reply, with a "?" or "!" before them.
  function dropStrayQuotes(tail) {
    let depth = 0;
    let out = "";
    for (const ch of tail) {
      if (ch === "„") depth++;
      if (ch === "“") {
        if (depth === 0) {
          out = out.replace(/[?!]+$/u, "");
          continue;
        }
        depth--;
      }
      out += ch;
    }
    return out;
  }

  function formatReply(promptSuffix, generated, reason) {
    const head = replyHead(promptSuffix);
    let text = normalize(head + dropStrayQuotes(generated)).replace(/\s+([.,;])/gu, "$1").trim();
    // sentence ends written by the model (not inside the quoted message, not abbreviations)
    const ends = [];
    const re = /[.;!?](?=\s|$)/gu;
    let m;
    while ((m = re.exec(text))) {
      if (m.index < head.length) continue;
      const nextStartsSentence = /^\s+\p{Lu}/u.test(text.slice(m.index + 1));
      if (text[m.index] === "." && !nextStartsSentence && isAbbreviation(text.slice(0, m.index))) continue;
      ends.push(m.index + 1);
    }
    if (ends.length >= 2 && ends[1] - ends[0] < 160) text = text.slice(0, ends[1]);
    else if (ends.length >= 1) text = text.slice(0, ends[0]);
    else if (reason !== "end") {
      const comma = text.lastIndexOf(",");
      text = (comma > head.length ? text.slice(0, comma) : text) + "…";
    }
    text = text.replace(/[;,]$/u, ".");
    if (!/[.!?…“"]$/u.test(text)) text += ".";
    return text;
  }

  const api = { sanitizeMessage, chatRequest, previewReply, formatReply, OPENINGS };
  root.TinyChat = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : globalThis);
