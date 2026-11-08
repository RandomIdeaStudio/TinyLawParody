// Prompt builder, mirror of tinylaw/prompt.py. The template comes from config.json.
(function (root) {
  "use strict";

  const CLAUSE = ["kterým", "jímž", "kterou", "jíž"];
  const PREPOSITIONS = ["o", "ze", "z", "ke", "k", "na", "pro"];

  function normalizeTitle(raw) {
    let s = String(raw).replace(/\s+/gu, " ").trim().replace(/\.+$/u, "").trim();
    s = s.replace(/^zákon(?![\p{L}\p{N}_])[,]?\s*/iu, "");
    if (!s) s = "o ničem";
    const first = s.split(" ", 1)[0].toLowerCase();
    if (CLAUSE.includes(first)) {
      const subject = first + s.slice(first.length);
      return ["Zákon, " + subject, subject];
    }
    let subject;
    if (!PREPOSITIONS.includes(first)) subject = "o " + s[0].toLowerCase() + s.slice(1);
    else subject = first + s.slice(first.length);
    return ["Zákon " + subject, subject];
  }

  function fill(template, values) {
    return template.replace(/\{(\w+)\}/g, (m, k) => (k in values ? String(values[k]) : m));
  }

  function buildPrompt(promptConfig, rawTitle, opts) {
    const o = Object.assign({ number: 1, year: 2026, day: 1, month: 4, part: 1 }, opts || {});
    const [title, subject] = normalizeTitle(rawTitle);
    return fill(promptConfig.template, {
      number: o.number,
      year: o.year,
      title: title,
      subject: subject,
      part: o.part,
      date: o.day + ". " + o.month + ". " + o.year,
      date_words: o.day + ". " + promptConfig.months_genitive[o.month - 1] + " " + o.year,
    });
  }

  const api = { normalizeTitle, buildPrompt };
  root.TinyPrompt = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : globalThis);
