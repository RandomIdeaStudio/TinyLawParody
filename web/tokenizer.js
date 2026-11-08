// Byte-level BPE tokenizer, mirror of tinylaw/tokenizer.py (GPT-2 style).
// Works in a Web Worker (importScripts) and in Node (require) for the parity test.
(function (root) {
  "use strict";

  function bytesToUnicode() {
    const bs = [];
    for (let b = 33; b <= 126; b++) bs.push(b);
    for (let b = 161; b <= 172; b++) bs.push(b);
    for (let b = 174; b <= 255; b++) bs.push(b);
    const cs = bs.slice();
    let n = 0;
    for (let b = 0; b < 256; b++) {
      if (bs.indexOf(b) < 0) {
        bs.push(b);
        cs.push(256 + n);
        n++;
      }
    }
    const enc = new Array(256);
    const dec = new Map();
    for (let i = 0; i < bs.length; i++) {
      const ch = String.fromCharCode(cs[i]);
      enc[bs[i]] = ch;
      dec.set(ch, bs[i]);
    }
    return { enc, dec };
  }

  // GPT-2 pre-tokenisation pattern; must match SPLIT_PATTERN in Python.
  const SPLIT = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

  class BPETokenizer {
    constructor(data) {
      this.vocab = data.vocab;
      this.eotId = data.eot_id;
      this.special = data.special_tokens || {};
      this.toId = new Map();
      for (let i = 0; i < this.vocab.length; i++) {
        if (this.vocab[i] !== null) this.toId.set(this.vocab[i], i);
      }
      this.ranks = new Map();
      for (let i = 0; i < data.merges.length; i++) this.ranks.set(data.merges[i], i);
      const m = bytesToUnicode();
      this.byteEnc = m.enc;
      this.byteDec = m.dec;
      this.cache = new Map();
      this.utf8 = new TextEncoder();
      this.utf8dec = new TextDecoder("utf-8", { fatal: false });
    }

    bpe(word) {
      const cached = this.cache.get(word);
      if (cached) return cached;
      let parts = Array.from(word);
      while (parts.length > 1) {
        let best = -1;
        let bestRank = Infinity;
        for (let i = 0; i < parts.length - 1; i++) {
          const r = this.ranks.get(parts[i] + " " + parts[i + 1]);
          if (r !== undefined && r < bestRank) {
            bestRank = r;
            best = i;
          }
        }
        if (best < 0) break;
        const a = parts[best];
        const b = parts[best + 1];
        const merged = [];
        for (let i = 0; i < parts.length; i++) {
          if (i < parts.length - 1 && parts[i] === a && parts[i + 1] === b) {
            merged.push(a + b);
            i++;
          } else {
            merged.push(parts[i]);
          }
        }
        parts = merged;
      }
      const ids = parts.map((p) => {
        const id = this.toId.get(p);
        if (id === undefined) throw new Error("token not in vocabulary: " + p);
        return id;
      });
      if (this.cache.size < 50000) this.cache.set(word, ids);
      return ids;
    }

    encodeOrdinary(text) {
      const ids = [];
      const pieces = text.match(SPLIT) || [];
      for (const piece of pieces) {
        let mapped = "";
        for (const byte of this.utf8.encode(piece)) mapped += this.byteEnc[byte];
        for (const id of this.bpe(mapped)) ids.push(id);
      }
      return ids;
    }

    encode(text) {
      const names = Object.keys(this.special);
      if (names.length === 0) return this.encodeOrdinary(text);
      const ids = [];
      let rest = text;
      while (rest.length) {
        let pos = -1;
        let name = null;
        for (const n of names) {
          const p = rest.indexOf(n);
          if (p >= 0 && (pos < 0 || p < pos)) {
            pos = p;
            name = n;
          }
        }
        if (pos < 0) {
          for (const id of this.encodeOrdinary(rest)) ids.push(id);
          break;
        }
        for (const id of this.encodeOrdinary(rest.slice(0, pos))) ids.push(id);
        ids.push(this.special[name]);
        rest = rest.slice(pos + name.length);
      }
      return ids;
    }

    // Raw bytes for token ids (special tokens give their text).
    tokenBytes(ids) {
      const out = [];
      for (const id of ids) {
        const tok = this.vocab[id];
        if (tok === null || tok === undefined) continue;
        if (Object.prototype.hasOwnProperty.call(this.special, tok)) {
          for (const b of this.utf8.encode(tok)) out.push(b);
          continue;
        }
        for (const ch of tok) out.push(this.byteDec.get(ch));
      }
      return new Uint8Array(out);
    }

    decode(ids) {
      return this.utf8dec.decode(this.tokenBytes(ids));
    }
  }

  const api = { BPETokenizer, bytesToUnicode };
  root.TinyTokenizer = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : globalThis);
