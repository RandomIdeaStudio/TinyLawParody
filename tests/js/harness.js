// Node helper for the Python parity tests. Reads a JSON request on stdin, writes JSON.
//   {"cmd": "encode", "tokenizer": path, "texts": [...]}
//   {"cmd": "decode", "tokenizer": path, "ids": [[...], ...]}
//   {"cmd": "logits", "model_dir": path, "ids": [...]}        -> logits at every position
//   {"cmd": "prompt", "config": path, "cases": [[title, opts], ...]}
"use strict";
const fs = require("fs");
const path = require("path");
const web = path.join(__dirname, "..", "..", "web");
const { BPETokenizer } = require(path.join(web, "tokenizer.js"));
const { TinyGPT } = require(path.join(web, "transformer.js"));
const { buildPrompt } = require(path.join(web, "prompt.js"));

function readJSON(p) {
  return JSON.parse(fs.readFileSync(p, "utf8"));
}

const req = JSON.parse(fs.readFileSync(0, "utf8"));
let out;
if (req.cmd === "encode") {
  const tok = new BPETokenizer(readJSON(req.tokenizer));
  out = req.texts.map((t) => tok.encode(t));
} else if (req.cmd === "decode") {
  const tok = new BPETokenizer(readJSON(req.tokenizer));
  out = req.ids.map((ids) => tok.decode(ids));
} else if (req.cmd === "logits") {
  const config = readJSON(path.join(req.model_dir, "config.json"));
  const buf = fs.readFileSync(path.join(req.model_dir, "model.bin"));
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
  const model = new TinyGPT(config, ab);
  out = req.ids.map((id) => Array.from(model.step(id)));
} else if (req.cmd === "prompt") {
  const config = readJSON(req.config);
  out = req.cases.map(([title, opts]) => buildPrompt(config.prompt, title, opts));
} else {
  throw new Error("unknown cmd " + req.cmd);
}
process.stdout.write(JSON.stringify(out));
