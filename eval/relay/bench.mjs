#!/usr/bin/env node
/**
 * relay-bench TTS phase — first-cut daemon replay bench (LOCAL approximation).
 *
 * Replays chat-shaped lines at a running `tts serve` daemon via the relay's
 * zero-dep client (relay/tts-client.js) and measures what the bot would feel:
 *   per line: ttfa_ms, wall_ms, audio_s, rtf, mode (cold|warm), chunks, rss_after_kb
 *   aggregate: ttfa med/p95, wall med, audio/wall throughput, cold-vs-warm delta
 *
 * GOVERNING LAW (T014 done-condition 6): the *real* relay-bench row must be
 * measured IN-CONTAINER (3.5 CPU / 2 GB cap) against the kokoroWorker
 * pipeline on the same lines. This harness is the engine-side half of that
 * comparison: run it against the daemon inside the container to produce the
 * admissible row. Locally it only approximates (taskset the daemon and this
 * script to ~4 cores: `taskset -c 0-3 tts serve …` etc.) and MUST be labeled
 * non-admissible.
 *
 * Usage:
 *   node bench.mjs --sock /tmp/tts/tts.sock --lane kokoro [--lane piper]
 *        [--workers N=1] [--reps R=1] [--seed 42] [--out eval/relay/results/<stamp>]
 *        [--lang en]
 *
 * Output: <out>/bench-lanes.tsv  (one row per line replay)
 *         <out>/summary.json     (per-lane aggregates + env stamp)
 * Zero dependencies; ESM; runs under the same Node as the relay bot.
 */
import { mkdirSync, writeFileSync, statSync } from "node:fs";
import path from "node:path";
import { daemonSynthesize, daemonAvailable } from "../../relay/tts-client.js";

// ---- chat-shaped corpus (what the bot actually reads aloud: 3–40 words,
// ---- speaker prefix on some lines, labeled link/emoji tokens, punctuation).
const CHAT_LINES = [
  "yo did anyone fix the build yet",
  "Riley: thanks, worked for me",
  "pushing now, give me ten minutes",
  "can you drop the link to the pull request",
  "i think that's the wrong commit, the diff looks inverted",
  "Mira: ok logging in, one sec",
  "lol that emoji made my entire day",
  "deploy's green, rollout finished about an hour ago",
  "Riley: which engine row did you end up trusting for the eval?",
  "hold on, my laptop just went to sleep mid-transfer, retrying",
  "just read the whole thread, i agree with the piper verdict",
  "Mira: the median looks fine but the p95 is what worries me",
  "ship it, worst case we roll back before the stream starts",
  "did the daemon come up in the container yet? still waiting here",
];

const args = process.argv.slice(2);
const opt = (name, dflt) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : dflt;
};
const SOCK = opt("sock", "/tmp/tts/tts.sock");
const LANES = args.flatMap((a, i) => (a === "--lane" ? [args[i + 1]] : []));
const WORKERS = parseInt(opt("workers", "1"), 10);
const REPS = parseInt(opt("reps", "1"), 10);
const SEED = parseInt(opt("seed", "42"), 10);
const LANG = opt("lang", "en");
const OUT = opt("out", "") || null;
const conn = { socket: SOCK };

if (!LANES.length) {
  console.error("usage: bench.mjs --sock PATH --lane LANE [--lane …] [--workers N] [--reps R] [--seed S] [--out DIR] [--lang L]");
  process.exit(2);
}

function summarize(times) {
  const s = [...times].sort((a, b) => a - b);
  const med = (p) => s[Math.min(s.length - 1, Math.floor((s.length - 1) * p))];
  return { med: med(0.5), p95: s[s.length - 1] };
}

async function replayLine(lane, text, rep) {
  const r = await daemonSynthesize(conn, {
    lane, text, lang: LANG, seed: SEED,
    timeoutMs: 120_000,
  });
  return {
    lane, rep, chars: text.length, text,
    mode: r.meta.mode ?? r.done.mode,
    ttfa_ms: Math.round(r.done.ttfa_ms), wall_ms: Math.round(r.done.wall_ms),
    audio_s: Number(r.done.audio_s?.toFixed(3)), rtf: Number(r.done.rtf?.toFixed(4)),
    chunks: r.done.chunks, rss_after_kb: r.done.rss_after_kb,
  };
}

const rows = [];
console.log(`relay-bench TTS phase (LOCAL approximation) — sock=${SOCK} workers=${WORKERS} reps=${REPS} seed=${SEED}`);
const ok = await daemonAvailable(conn);
if (!ok) {
  console.error(`daemon not reachable at ${SOCK}`);
  process.exit(4);
}
console.log(`health gate: ok (${new Date().toISOString()})`);

for (const lane of LANES) {
  // warm the lane once outside timing so rep/cold accounting is explicit
  const warm = await replayLine(lane, "warming the lane", 1);
  rows.push({ ...warm, phase: "warmup" });

  // workers ≥2: concurrent in-flight requests; daemon serializes per lane (worker loop=1)
  const jobs = [];
  for (let w = 0; w < WORKERS; w++) {
    const mine = CHAT_LINES.filter((_, i) => i % WORKERS === w);
    jobs.push(
      (async () => {
        for (let rep = 1; rep <= REPS; rep++)
          for (const text of mine) rows.push({ ...(await replayLine(lane, text, rep)), phase: "main" });
      })(),
    );
  }
  await Promise.all(jobs);
  const laneRows = rows.filter((r) => r.lane === lane && r.phase === "main");
  const tt = summarize(laneRows.map((r) => r.ttfa_ms));
  const wall = summarize(laneRows.map((r) => r.wall_ms));
  const totAudio = laneRows.reduce((n, r) => n + r.audio_s, 0);
  const totWall = laneRows.reduce((n, r) => n + r.wall_ms, 0) / 1000;
  console.log(
    `  ${lane}: n=${laneRows.length} ttfa med ${tt.med}ms / p95 ${tt.p95}ms · wall med ${wall.med}ms · rtf_mean ${(totAudio / totWall).toFixed(3)} · rss_max_kb ${Math.max(...laneRows.map((r) => r.rss_after_kb))}`,
  );
}

if (OUT) {
  mkdirSync(OUT, { recursive: true });
  const tsvPath = path.join(OUT, "bench-lanes.tsv");
  const cols = ["lane", "phase", "rep", "chars", "mode", "ttfa_ms", "wall_ms", "audio_s", "rtf", "chunks", "rss_after_kb"];
  const esc = (s) => `"${String(s).replaceAll('"', '""')}"`;
  writeFileSync(
    tsvPath,
    cols.join("\t") + "\n" + rows.map((r) => cols.map((c) => esc(r[c] ?? "")).join("\t")).join("\n") + "\n",
  );
  const per = {};
  for (const lane of LANES) {
    const main = rows.filter((r) => r.lane === lane && r.phase === "main");
    const cold = rows.find((r) => r.lane === lane && r.phase === "warmup");
    per[lane] = {
      n: main.length,
      ttfa: summarize(main.map((r) => r.ttfa_ms)),
      wall: summarize(main.map((r) => r.wall_ms)),
      rtf_mean: main.reduce((n, r) => n + r.audio_s, 0) / (main.reduce((n, r) => n + r.wall_ms, 0) / 1000),
      audio_total_s: Number(main.reduce((n, r) => n + r.audio_s, 0).toFixed(2)),
      rss_after_kb_max: Math.max(...main.map((r) => r.rss_after_kb)),
      warmup: cold ? { ttfa_ms: cold.ttfa_ms, rss_after_kb: cold.rss_after_kb } : null,
    };
  }
  writeFileSync(
    path.join(OUT, "summary.json"),
    JSON.stringify({ label: "local-approximation (non-admissible; in-container row required)", stamp: new Date().toISOString(), sock: SOCK, workers: WORKERS, reps: REPS, seed: SEED, lang: LANG, lines: CHAT_LINES.length, host: { platform: process.platform, node: process.version }, lanes: per }, null, 2) + "\n",
  );
  console.log(`wrote ${tsvPath} + ${path.join(OUT, "summary.json")}`);
}
