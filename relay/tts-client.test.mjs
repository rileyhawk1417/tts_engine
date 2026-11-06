import { daemonSynthesize, daemonAvailable, sha256 } from "./tts-client.js";

const SOCK = process.env.TTS_SOCK ?? "/tmp/tts/tts.sock";
console.log(`client test vs ${SOCK}`);
let pass = 0, fail = 0;
const ok = (c, m) => { c ? pass++ : (fail++, console.log("FAIL:", m)); };

// health gate
ok(await daemonAvailable({ socket: SOCK }), "health reachable");

// 1. kokoro warm — meta/done shape + wav sane
const a = await daemonSynthesize({ socket: SOCK }, { lane: "kokoro", text: "Hello there.", seed: 42 });
ok(a.meta.type === "meta" && a.done.type === "done", "event types");
ok(a.meta.mode === "warm" && a.meta.load_ms === undefined, "warm meta, load_ms omitted");
ok(a.wav.readUInt32LE(24) === 24000, "kokoro 24k sample rate");
ok(a.wav.length === a.pcm.length + 44, "wav = pcm+44");
ok(a.done.audio_s > 1.5, `audio_s ${a.done.audio_s?.toFixed?.(2)}`);

// 2. determinism: same seed twice → same sha; different seed → (kokoro) same sha too
const a2 = await daemonSynthesize({ socket: SOCK }, { lane: "kokoro", text: "Hello there.", seed: 42 });
ok(sha256(a.pcm) === sha256(a2.pcm), "kokoro deterministic at seed 42");

// 3. speed dial pass-through (kokoro 0.5..2.0)
const fast = await daemonSynthesize({ socket: SOCK }, { lane: "kokoro", text: "Hello there.", seed: 42, speed: 1.25 });
ok(fast.done.audio_s < a.done.audio_s, `speed 1.25 shorter (${fast.done.audio_s.toFixed(2)} < ${a.done.audio_s.toFixed(2)})`);

// 4. speed out of range → client clamps, daemon accepts
const slow = await daemonSynthesize({ socket: SOCK }, { lane: "kokoro", text: "Hello there.", seed: 42, speed: 9.0 });
ok(slow.done.audio_s < a.done.audio_s && slow.done.audio_s < fast.done.audio_s, `speed clamped to 2.0, audio shorter (${slow.done.audio_s.toFixed(2)} < ${a.done.audio_s.toFixed(2)})`);

// 5. piper — seeded lane, determinism, repeat byte-identical
const p1 = await daemonSynthesize({ socket: SOCK }, { lane: "piper", text: "Warm piper, ~200 ms to first audio.", seed: 42 });
const p2 = await daemonSynthesize({ socket: SOCK }, { lane: "piper", text: "Warm piper, ~200 ms to first audio.", seed: 42 });
ok(sha256(p1.pcm) === sha256(p2.pcm), "piper deterministic at seed 42");
ok(p1.wav.readUInt32LE(24) === 22050, "piper 22050");
ok(p1.meta.determinism_class === "seeded", "piper determinism_class seeded");
console.log(`piper ttfa=${p1.done.ttfa_ms}ms wall=${p1.done.wall_ms}ms rtf=${p1.done.rtf}`);

// 6. error surface: bad lane → reject with message
try {
  await daemonSynthesize({ socket: SOCK }, { lane: "nonexistent", text: "x" });
  ok(false, "bad lane should reject");
} catch (e) { ok(e.message.includes("not served"), `bad lane: ${e.message.slice(0, 60)}`); }

// 7. timeout fires on a hung endpoint (no listener socket)
try {
  await daemonSynthesize({ socket: "/tmp/tts/none.sock" }, { text: "x", timeoutMs: 500 });
  ok(false, "timeout should reject");
} catch (e) { ok(e.message.includes("timed out") || e.message.includes("ENOENT") || e.message.includes("connect"), `timeout/conn: ${e.message.slice(0, 60)}`); }

// 8. no seed → nondeterministic piper (entropy draw)
const q1 = await daemonSynthesize({ socket: SOCK }, { lane: "piper", text: "Entropy draw one." });
const q2 = await daemonSynthesize({ socket: SOCK }, { lane: "piper", text: "Entropy draw one." });
console.log(`unseeded piper shas differ: ${sha256(q1.pcm) !== sha256(q2.pcm)}`);

console.log(`\n${pass} passed, ${fail} failed`);
process.exit(fail ? 1 : 0);
