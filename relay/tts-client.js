/**
 * tts daemon client — the zero-dependency relay-side `TtsEngine` for
 * `tts serve` (tts_engine T014). Paste-ready for `bot-relay/src/tts.ts`
 * (plain ESM JavaScript + JSDoc: valid TypeScript, runs under Node unchanged).
 *
 * Wire (docs/notes/resident-service.md, "the wire, as landed"):
 *   POST /synth  body {lane, text, lang, voice, seed, speed}
 *     → 200, HTTP-chunked Transfer-Encoding, events split on '\n' INSIDE payloads:
 *        {type:"meta",…} → {type:"chunk",seq,pcm_b64,bytes,from}×N → {type:"done",…}
 *   GET /health  → {lanes:[{lane,loaded,…}], …}   (used for availability gate)
 *
 * The client owns: WAV assembly from LE mono i16 PCM, per-lane `speed` clamps
 * (the daemon hard-errors 400 out of range), a hard request timeout, and an
 * availability cache for the health gate. It does NOT own chunk planning —
 * send the full text; the daemon chunks at sentences.
 *
 * Node's builtin http decodes chunked framing over a UDS via `socketPath`
 * (TCP via host/port) — no dependencies, matching relay's zero-dep policy.
 *
 * @typedef {Object} DaemonEndpoint UDS: {socket:"/run/tts/tts.sock"} · TCP: {host,port}
 * @typedef {Object} DaemonSynthArgs
 * @property {string} [lane] lane name as served (omit on a single-lane daemon)
 * @property {string} text message text (the daemon chunks it)
 * @property {string} [lang] "en"|"fr"|"de"|"es"|"pt" (default en)
 * @property {string} [voice] voice id (kokoro: e.g. "af_heart") — path or embedding
 * @property {number} [seed] set for byte-reproducible responses
 * @property {number} [speed] per-lane rate dial (clamped client-side, see SPEED_CLAMP)
 * @property {number} [timeoutMs] hard request timeout (default 30000)
 * @typedef {Object} DaemonSynthResult
 * @property {Buffer} wav assembled WAV (mono 16-bit at the lane rate)
 * @property {Buffer} pcm raw concatenated LE mono i16 PCM
 * @property {{type:string,[k:string]:any}} meta first event (mode/load_ms/…)
 * @property {{type:string,[k:string]:any}} done last event (ttfa_ms/wall_ms/rtf/…)
 */
import http from "node:http";
import crypto from "node:crypto";

/** sha256 hex of a buffer (determinism spot-checks, bench identity). */
export const sha256 = (buf) => crypto.createHash("sha256").update(buf).digest("hex");

/**
 * Per-lane `speed` clamps the daemon enforces (docs/usage.md §3.4). The daemon
 * hard-errors 400 out of range, so clamp here; null → field omitted.
 * lane → {min,max,default}; lanes with no dial are simply omitted.
 */
export const SPEED_CLAMP = {
  kokoro: { min: 0.5, max: 2.0, def: null },
  piper: { min: 0.25, max: 4.0, def: null },
};

/**
 * @param {DaemonEndpoint} conn
 * @param {DaemonSynthArgs} req
 * @returns {Promise<DaemonSynthResult>}
 */
export function daemonSynthesize(conn, req) {
  const body = {};
  if (req.lane) body.lane = req.lane;
  if (!req.text || !req.text.trim()) throw new Error("daemon synth: empty text");
  body.text = req.text;
  if (req.lang) body.lang = req.lang;
  if (req.voice) body.voice = req.voice;
  if (req.seed !== undefined && req.seed !== null) body.seed = req.seed;
  if (req.speed !== undefined && req.speed !== null) {
    const sc = SPEED_CLAMP[req.lane];
    const sp = Math.min(sc.max, Math.max(sc.min, req.speed));
    body.speed = Number.isInteger(sp) ? sp : Number(sp.toFixed(4));
  }
  const payload = JSON.stringify(body);
  const timeoutMs = req.timeoutMs ?? 30000;

  return new Promise((resolve, reject) => {
    /** @type {Buffer[]} */
    const bufs = [];
    const r = http.request(
      {
        socketPath: conn.socket,
        host: conn.host,
        port: conn.port,
        path: "/synth",
        method: "POST",
        headers: { "content-type": "application/json" },
      },
      (res) => {
        res.on("error", (e) =>
          reject(new Error(`daemon /synth stream error: ${e.message}`)),
        );
        res.on("data", (b) => bufs.push(b));
        res.on("end", () => {
          clearTimeout(timer);
          /** @type {{type:string,[k:string]:any}[]} */
          let events = [];
          try {
            events = Buffer.concat(bufs)
              .toString("utf8")
              .split("\n")
              .filter(Boolean)
              .map((l) => JSON.parse(l));
          } catch (e) {
            return reject(new Error(`daemon: bad ndjson (${e.message})`));
          }
          const errEvent = events.find((e) => e.type === "error");
          if (res.statusCode !== 200 || errEvent) {
            return reject(
              new Error(
                `daemon /synth ${res.statusCode}: ${errEvent?.message ?? "no events"}`,
              ),
            );
          }
          const meta = events[0];
          if (!meta || meta.type !== "meta")
            return reject(new Error("daemon: first event not meta"));
          const done = events[events.length - 1];
          if (done.type !== "done") return reject(new Error("daemon: last event not done"));
          const pcm = Buffer.concat(
            events
              .filter((e) => e.type === "chunk")
              .map((e) => Buffer.from(e.pcm_b64, "base64")),
          );
          resolve({ wav: wavHeader(pcm, meta.sample_rate), pcm, meta, done });
        });
      },
    );
    const timer = setTimeout(() => {
      r.destroy(new Error(`daemon /synth timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    r.on("error", (e) => {
      clearTimeout(timer);
      reject(e);
    });
    r.end(payload);
  });
}

let _healthCache = { ok: false, at: 0 };

/**
 * Availability gate: GET /health over the same endpoint, cached `cacheMs`
 * (default 5 s) so hot paths don't probe per message.
 * @param {DaemonEndpoint} conn
 * @param {{timeoutMs?: number, cacheMs?: number}} [opts]
 * @returns {Promise<boolean>}
 */
export function daemonAvailable(conn, opts = {}) {
  const cacheMs = opts.cacheMs ?? 5000;
  const fresh = Date.now() - _healthCache.at < cacheMs;
  if (fresh) return Promise.resolve(_healthCache.ok);
  const bad = () => {
    _healthCache = { ok: false, at: Date.now() };
    return false;
  };
  return new Promise((resolve) => {
    const r = http.request(
      { socketPath: conn.socket, host: conn.host, port: conn.port, path: "/health", method: "GET" },
      (res) => {
        res.resume();
        res.on("end", () => {
          _healthCache = { ok: res.statusCode === 200, at: Date.now() };
          resolve(_healthCache.ok);
        });
      },
    );
    r.setTimeout(opts.timeoutMs ?? 2000, () => {
      r.destroy();
      resolve(bad());
    });
    r.on("error", () => resolve(bad()));
    r.end();
  });
}

/** Canonical 44-byte WAV header + data (mono, 16-bit little-endian). */
export function wavHeader(pcm, sampleRate) {
  const hdr = Buffer.alloc(44);
  hdr.write("RIFF", 0);
  hdr.writeUInt32LE(36 + pcm.length, 4);
  hdr.write("WAVE", 8);
  hdr.write("fmt ", 12);
  hdr.writeUInt32LE(16, 16);
  hdr.writeUInt16LE(1, 20); // PCM
  hdr.writeUInt16LE(1, 22); // mono
  hdr.writeUInt32LE(sampleRate, 24);
  hdr.writeUInt32LE(sampleRate * 2, 28); // byte rate
  hdr.writeUInt16LE(2, 32); // block align
  hdr.writeUInt16LE(16, 34);
  hdr.write("data", 36);
  hdr.writeUInt32LE(pcm.length, 40);
  return Buffer.concat([hdr, pcm]);
}
