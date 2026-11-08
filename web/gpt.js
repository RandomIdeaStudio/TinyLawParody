// Forward pass of the tiny GPT in plain JavaScript (float32 math, float16 weights).
// Mirror of tinylaw/model.py. Works in a Web Worker and in Node (parity test).
(function (root) {
  "use strict";

  // float16 -> float32 lookup table
  let HALF = null;
  function halfTable() {
    if (HALF) return HALF;
    HALF = new Float32Array(65536);
    for (let h = 0; h < 65536; h++) {
      const s = h & 0x8000 ? -1 : 1;
      const e = (h >> 10) & 0x1f;
      const f = h & 0x3ff;
      let v;
      if (e === 0) v = s * Math.pow(2, -14) * (f / 1024);
      else if (e === 31) v = f ? NaN : s * Infinity;
      else v = s * Math.pow(2, e - 15) * (1 + f / 1024);
      HALF[h] = v;
    }
    return HALF;
  }

  function loadWeights(config, buffer) {
    const table = halfTable();
    const u16 = new Uint16Array(buffer);
    const w = {};
    for (const t of config.tensors) {
      const start = t.offset / 2;
      const arr = new Float32Array(t.length);
      for (let i = 0; i < t.length; i++) arr[i] = table[u16[start + i]];
      w[t.name] = arr;
    }
    return w;
  }

  // y[o] = b[o] + sum_i x[i] * W[o, i]   (W stored row-major [out, in], as in PyTorch)
  function linear(x, W, b, inDim, outDim, y) {
    for (let o = 0; o < outDim; o++) {
      let s = b ? b[o] : 0;
      const row = o * inDim;
      for (let i = 0; i < inDim; i++) s += x[i] * W[row + i];
      y[o] = s;
    }
    return y;
  }

  function layerNorm(x, g, b, n, eps, y) {
    let mean = 0;
    for (let i = 0; i < n; i++) mean += x[i];
    mean /= n;
    let v = 0;
    for (let i = 0; i < n; i++) {
      const d = x[i] - mean;
      v += d * d;
    }
    const inv = 1 / Math.sqrt(v / n + eps);
    for (let i = 0; i < n; i++) y[i] = (x[i] - mean) * inv * g[i] + b[i];
    return y;
  }

  const GELU_C = Math.sqrt(2 / Math.PI);
  function gelu(x) {
    return 0.5 * x * (1 + Math.tanh(GELU_C * (x + 0.044715 * x * x * x)));
  }

  class TinyGPT {
    constructor(config, buffer) {
      this.cfg = config.model;
      this.w = loadWeights(config, buffer);
      const c = this.cfg;
      const C = c.n_embd;
      this.kCache = [];
      this.vCache = [];
      for (let l = 0; l < c.n_layer; l++) {
        this.kCache.push(new Float32Array(c.context * C));
        this.vCache.push(new Float32Array(c.context * C));
      }
      this.pos = 0;
      this.x = new Float32Array(C);
      this.h = new Float32Array(C);
      this.qkv = new Float32Array(3 * C);
      this.att = new Float32Array(C);
      this.tmp = new Float32Array(C);
      this.ff = new Float32Array(4 * C);
      this.scores = new Float32Array(c.context);
      this.logits = new Float32Array(c.vocab_size);
    }

    reset() {
      this.pos = 0;
    }

    // Feed one token at the next position; returns logits for the next token.
    // The returned array is reused between calls.
    step(token) {
      const c = this.cfg;
      const C = c.n_embd;
      const H = c.n_head;
      const hs = C / H;
      const eps = c.layer_norm_eps;
      if (this.pos >= c.context) throw new Error("context full");
      const w = this.w;
      const x = this.x;
      const te = w["wte.weight"];
      const pe = w["wpe.weight"];
      const t = this.pos;
      for (let i = 0; i < C; i++) x[i] = te[token * C + i] + pe[t * C + i];
      const scale = 1 / Math.sqrt(hs);
      for (let l = 0; l < c.n_layer; l++) {
        const p = "blocks." + l + ".";
        layerNorm(x, w[p + "ln1.weight"], w[p + "ln1.bias"], C, eps, this.h);
        linear(this.h, w[p + "qkv.weight"], w[p + "qkv.bias"], C, 3 * C, this.qkv);
        const K = this.kCache[l];
        const V = this.vCache[l];
        for (let i = 0; i < C; i++) {
          K[t * C + i] = this.qkv[C + i];
          V[t * C + i] = this.qkv[2 * C + i];
        }
        for (let hd = 0; hd < H; hd++) {
          const off = hd * hs;
          let max = -Infinity;
          for (let j = 0; j <= t; j++) {
            let s = 0;
            const kb = j * C + off;
            for (let i = 0; i < hs; i++) s += this.qkv[off + i] * K[kb + i];
            s *= scale;
            this.scores[j] = s;
            if (s > max) max = s;
          }
          let sum = 0;
          for (let j = 0; j <= t; j++) {
            const e = Math.exp(this.scores[j] - max);
            this.scores[j] = e;
            sum += e;
          }
          for (let i = 0; i < hs; i++) this.att[off + i] = 0;
          for (let j = 0; j <= t; j++) {
            const a = this.scores[j] / sum;
            const vb = j * C + off;
            for (let i = 0; i < hs; i++) this.att[off + i] += a * V[vb + i];
          }
        }
        linear(this.att, w[p + "proj.weight"], w[p + "proj.bias"], C, C, this.tmp);
        for (let i = 0; i < C; i++) x[i] += this.tmp[i];
        layerNorm(x, w[p + "ln2.weight"], w[p + "ln2.bias"], C, eps, this.h);
        linear(this.h, w[p + "fc.weight"], w[p + "fc.bias"], C, 4 * C, this.ff);
        for (let i = 0; i < 4 * C; i++) this.ff[i] = gelu(this.ff[i]);
        linear(this.ff, w[p + "fc_out.weight"], w[p + "fc_out.bias"], 4 * C, C, this.tmp);
        for (let i = 0; i < C; i++) x[i] += this.tmp[i];
      }
      layerNorm(x, w["ln_f.weight"], w["ln_f.bias"], C, eps, this.h);
      linear(this.h, te, null, C, c.vocab_size, this.logits); // tied output head
      this.pos++;
      return this.logits;
    }

    // Feed a whole sequence from position 0; returns a copy of the last logits.
    prefill(tokens) {
      this.reset();
      let logits = null;
      for (const tok of tokens) logits = this.step(tok);
      return logits ? Float32Array.from(logits) : null;
    }
  }

  // Small seeded random generator (mulberry32).
  function rng(seed) {
    let a = seed >>> 0;
    return function () {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // Temperature + top-k sampling.
  function sample(logits, temperature, topK, random) {
    const n = logits.length;
    const k = Math.max(1, Math.min(topK || n, n));
    const idx = new Int32Array(n);
    for (let i = 0; i < n; i++) idx[i] = i;
    const top = Array.from(idx).sort((a, b) => logits[b] - logits[a]).slice(0, k);
    const t = Math.max(temperature, 1e-5);
    const max = logits[top[0]];
    const probs = top.map((i) => Math.exp((logits[i] - max) / t));
    const sum = probs.reduce((a, b) => a + b, 0);
    let r = random() * sum;
    for (let i = 0; i < k; i++) {
      r -= probs[i];
      if (r <= 0) return top[i];
    }
    return top[k - 1];
  }

  const api = { TinyGPT, sample, rng, halfTable };
  root.TinyGPTLib = api;
  if (typeof module !== "undefined" && module.exports) module.exports = api;
})(typeof self !== "undefined" ? self : globalThis);
