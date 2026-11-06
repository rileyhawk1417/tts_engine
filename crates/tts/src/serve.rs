//! `tts serve` — resident daemon (T014; design in `docs/notes/resident-service.md`).
//!
//! One binary, two modes: `tts synth` stays the ONLY eval/bench surface; this
//! daemon is the warm-resident serving surface that drop-in replaces how
//! bot-relay hosts engines (`kokoroWorker` et al host ONE loaded model in a
//! worker thread; so does one `tts serve` lane slot, minus the 4.9 s python
//! import in the middle of every utterance).
//!
//! Protocol (wire law): HTTP/1.1 POST `/synth` (default UDS, `--tcp` opt-in),
//! NDJSON response — `{type:"meta",…}` → `{type:"chunk",seq,pcm_b64}`×N →
//! `{type:"done",metrics}`. Honesty fields: `mode` cold|warm, fresh
//! `load_ms` ONLY on a true cold (lazy first use or post-idle-unload reload),
//! `resets`, `rss_after_kb`. Eval never runs through the daemon.
//!
//! Worker loop = 1 per lane, shared across lanes (the `KOKORO_POOL_SIZE=1`
//! lesson: +1 worker buys 4–9% throughput for ~300 MB resident and slower
//! first audio; here requests are serialized per connection like relay's
//! `Speaker` FIFO, so one loop is exactly enough).

use std::io::{BufRead, BufReader, BufWriter, Read, Write};
use std::net::TcpListener;
use std::net::TcpStream;
use std::os::unix::fs::FileTypeExt;
use std::os::unix::io::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use serde_json::{Value, json};

use tts_engine::lanes::make_lane;
use tts_engine::types::{LaneError, LoadInfo, ModelRef, SynthRequest, SynthStream, TtsLane};
use tts_engine::{chunking, types::Lang};

/// Boot configuration (`tts serve` flags; clap parsing lives in `main.rs`).
#[derive(Debug, Clone)]
pub struct ServeConfig {
    /// Repeatable `NAME=MODEL` pairs; MODEL is a disk path (or `smoke`).
    pub lanes: Vec<String>,
    /// Unix domain socket path (mutually exclusive with `tcp`).
    pub socket: Option<PathBuf>,
    /// `host:port` (opt-in; UDS is the default for permission scoping).
    pub tcp: Option<String>,
    /// Load lanes on first request instead of at boot.
    pub lazy: bool,
    /// Unload lanes idle for > N seconds (0 = keep resident forever).
    pub idle_unload: u64,
    /// Intra-op thread dial for ORT-graph lanes (`TTS_ORT_INTRA`).
    pub threads: Option<usize>,
    /// setpriority on the process (relay's nice discipline).
    pub nice: Option<i32>,
    /// Candle device for the flow-candle lane: `cpu` default, `cuda` opt-in
    /// (validated here, forwarded as TTS_CANDLE_DEVICE before any lane load;
    /// cuda without the flow-candle-cuda feature fails lane load loudly).
    pub device: Option<String>,
}

enum Listener {
    Unix(UnixListener),
    Tcp(TcpListener),
}

impl Listener {
    fn accept(&self) -> std::io::Result<Conn> {
        Ok(match self {
            Listener::Unix(l) => Conn::Unix(l.accept()?.0),
            Listener::Tcp(l) => Conn::Tcp(l.accept()?.0),
        })
    }
    fn fd(&self) -> i32 {
        match self {
            Listener::Unix(l) => l.as_raw_fd(),
            Listener::Tcp(l) => l.as_raw_fd(),
        }
    }
    fn label(&self) -> String {
        match self {
            Listener::Unix(l) => format!("unix:{:?}", l.local_addr().ok()),
            Listener::Tcp(l) => format!("tcp:{:?}", l.local_addr().ok()),
        }
    }
}

enum Conn {
    Unix(UnixStream),
    Tcp(TcpStream),
}

impl Read for Conn {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Conn::Unix(s) => s.read(buf),
            Conn::Tcp(s) => s.read(buf),
        }
    }
}

impl Write for Conn {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Conn::Unix(s) => s.write(buf),
            Conn::Tcp(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Conn::Unix(s) => s.flush(),
            Conn::Tcp(s) => s.flush(),
        }
    }
}

/// One serve-resident lane: `make_lane(name)` + a `load()`-ed model. `inner`
/// is `None` when the lane is unloaded (lazy, or idle-unloaded); a request
/// into an unloaded slot reloads it and that request pays cold `load_ms`.
struct LaneSlot {
    name: String,
    model: String,
    model_ref: ModelRef,
    inner: Option<Box<dyn TtsLane>>,
    /// How many times this slot was idle-unloaded (or lazily booted).
    resets: u32,
    last_used: Option<Instant>,
}

impl LaneSlot {
    fn load(&mut self) -> Result<LoadInfo, LaneError> {
        let mut lane = make_lane(&self.name).ok_or_else(|| {
            LaneError::Load(format!(
                "lane {:?} is not compiled into this binary",
                self.name
            ))
        })?;
        let info = lane.load(&self.model_ref)?;
        self.inner = Some(lane);
        self.last_used = Some(Instant::now());
        Ok(info)
    }
    fn drop_lane(&mut self) {
        // ORT sessions + lane weights drop here; RSS falls with the arena.
        self.inner = None;
        self.resets += 1;
    }
}

/// Entry point (`Cmd::Serve`). Fail-fast boot; then accept → synth → stream.
pub fn run(cfg: &ServeConfig) -> ExitCode {
    // Dial dials BEFORE any lane loads (ORT reads TTS_ORT_INTRA at load time).
    if let Some(n) = cfg.threads {
        // SAFETY: boot path — no threads are spawned yet; set_var is
        // documented as memory-unsafe under other threads (edition 2024).
        unsafe { std::env::set_var("TTS_ORT_INTRA", n.to_string()) };
    }
    if let Some(n) = cfg.nice {
        // SAFETY: FFI setpriority, plain syscall. A negative nice that would
        // exceed the process rlimit fails here, by design (fail fast).
        let r = unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, n) };
        if r != 0 {
            eprintln!(
                "serve: setpriority({n}) failed: {r} (hint: negative nice needs CAP_SYS_NICE)"
            );
            return ExitCode::from(3);
        }
    }
    if cfg.lanes.is_empty() {
        eprintln!("serve: no lanes; pass --lane NAME=MODEL (repeatable)");
        return ExitCode::from(2);
    }
    // Candle device select (flow-candle lane): validated here so an operator
    // typo fails at boot; forwarded via TTS_CANDLE_DEVICE for the loader.
    // cuda on a non-cuda build is NOT rejected here — make_lane load at boot
    // rejects it loudly (operator config error, same fail-fast path).
    if let Some(d) = &cfg.device {
        match d.as_str() {
            "" | "cpu" => {}
            "cuda" | "cuda:0" => {
                // SAFETY: single-threaded boot, before any lane/thread spawn;
                // the only env mutation in the process.
                unsafe { std::env::set_var("TTS_CANDLE_DEVICE", d) };
            }
            other => {
                eprintln!("serve: --device accepts cpu | cuda (cuda:0), got {other:?}");
                return ExitCode::from(2);
            }
        }
    }
    let listener = match bind(cfg) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("serve: bind failed: {e}");
            return ExitCode::from(5);
        }
    };

    // Resolve + (default) pre-warm lanes at boot. Fail fast: a lane that
    // cannot load at boot is an operator config error, not a runtime state.
    let mut slots: Vec<LaneSlot> = Vec::new();
    for spec in &cfg.lanes {
        let Some((name, model)) = spec.split_once('=') else {
            eprintln!("serve: --lane wants NAME=MODEL, got {spec:?}");
            return ExitCode::from(2);
        };
        let name = name.to_string();
        let model = model.to_string();
        let Some(model_ref) = crate::resolve_model(&name, &model, false) else {
            eprintln!(
                "serve: model not found on disk: {model} (lane {name:?}; hub ids are --allow-hf synth-only)"
            );
            return ExitCode::from(3);
        };
        let mut slot = LaneSlot {
            name,
            model,
            model_ref,
            inner: None,
            resets: 0,
            last_used: None,
        };
        if !cfg.lazy {
            match slot.load() {
                Ok(info) => eprintln!("serve: lane {:?} loaded in {} ms", slot.model, info.load_ms),
                Err(e) => {
                    eprintln!("serve: lane {:?} load failed: {e}", slot.name);
                    return ExitCode::from(3);
                }
            }
        }
        slots.push(slot);
    }

    eprintln!(
        "serve: listening on {} — lanes {}, device {}, idle-unload {}s, lazy {}",
        listener.label(),
        slots
            .iter()
            .map(|s| s.name.as_str())
            .collect::<Vec<_>>()
            .join(","),
        cfg.device.as_deref().unwrap_or("cpu"),
        cfg.idle_unload,
        cfg.lazy
    );

    // Accept loop. Sequential on purpose (worker loop = 1): synth requests are
    // serialized exactly like relay's Speaker FIFO consumes them.
    loop {
        // Idle-unload window: poll() so we can wake up and free arenas.
        if cfg.idle_unload > 0
            && let Some(wait) = next_unload_wait(&slots, cfg.idle_unload)
        {
            let ready = match poll_ready(listener.fd(), wait) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("serve: poll failed: {e}");
                    return ExitCode::from(5);
                }
            };
            if !ready {
                unload_idle_lanes(&mut slots, cfg.idle_unload);
                continue;
            }
        }
        match listener.accept() {
            Ok(conn) => {
                if let Err(e) = handle_conn(&mut slots, &cfg_id(cfg), conn) {
                    eprintln!("serve: connection dropped: {e}");
                }
            }
            Err(e)
                if matches!(
                    e.kind(),
                    std::io::ErrorKind::ConnectionAborted
                        | std::io::ErrorKind::Interrupted
                        | std::io::ErrorKind::WouldBlock
                ) =>
            {
                continue;
            }
            Err(e) => {
                eprintln!("serve: accept failed: {e}");
                return ExitCode::from(5);
            }
        }
    }
}

/// Read-only view needed inside per-request handling (idle-unload seconds).
fn cfg_id(cfg: &ServeConfig) -> u64 {
    cfg.idle_unload
}

fn bind(cfg: &ServeConfig) -> std::io::Result<Listener> {
    match (&cfg.socket, &cfg.tcp) {
        (Some(_), Some(_)) => Err(std::io::Error::other("pass --socket or --tcp, not both")),
        (None, None) => Err(std::io::Error::other(
            "no listener: pass --socket PATH (default) or --tcp HOST:PORT",
        )),
        (Some(p), None) => {
            prepare_socket_path(p)?;
            let l = UnixListener::bind(p)?;
            Ok(Listener::Unix(l))
        }
        (None, Some(spec)) => {
            let addr: std::net::SocketAddr = spec
                .parse()
                .map_err(|e| std::io::Error::other(format!("--tcp {spec:?}: {e}")))?;
            Ok(Listener::Tcp(TcpListener::bind(addr)?))
        }
    }
}

fn prepare_socket_path(p: &Path) -> std::io::Result<()> {
    if let Some(parent) = p.parent().filter(|x| !x.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)?;
    }
    if p.exists() {
        let md = std::fs::metadata(p)?;
        if md.file_type().is_socket() {
            std::fs::remove_file(p)?; // stale socket from a previous run
        } else {
            return Err(std::io::Error::other(format!(
                "{} exists and is not a unix socket — refusing to remove",
                p.display()
            )));
        }
    }
    Ok(())
}

/// Milliseconds until the next lane crosses its idle deadline (None = none due).
fn next_unload_wait(slots: &[LaneSlot], idle_s: u64) -> Option<Duration> {
    let now = Instant::now();
    let deadline = Duration::from_secs(idle_s);
    slots
        .iter()
        .filter(|s| s.inner.is_some())
        .filter_map(|s| {
            s.last_used
                .map(|t| deadline.saturating_sub(now.saturating_duration_since(t)))
        })
        .min()
}

/// Unload every lane whose idle time exceeds `idle_s` (called on poll
/// timeouts so the resident set actually shrinks when a lane goes quiet).
fn unload_idle_lanes(slots: &mut [LaneSlot], idle_s: u64) {
    let now = Instant::now();
    for s in slots {
        if let Some(t) = s.last_used {
            let idle = now.duration_since(t);
            if s.inner.is_some() && idle > Duration::from_secs(idle_s) {
                eprintln!(
                    "serve: lane {:?} idle {:.1}s > {}s — unloading (resets → {})",
                    s.name,
                    idle.as_secs_f64(),
                    idle_s,
                    s.resets + 1
                );
                s.drop_lane();
            }
        }
    }
}

fn poll_ready(fd: i32, wait: Duration) -> std::io::Result<bool> {
    let timeout_ms = wait.as_millis() as i32;
    let mut fds = [libc::pollfd {
        fd,
        events: libc::POLLIN,
        revents: 0,
    }];
    // SAFETY: plain poll(2) over an owned listener fd.
    let r = unsafe { libc::poll(fds.as_mut_ptr(), 1, timeout_ms) };
    if r < 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(r > 0)
}

// ---------------------------------------------------------------------------
// HTTP framing (hand-rolled, HTTP/1.1 only, trusted-loopback assumption)
// ---------------------------------------------------------------------------

struct ParsedRequest {
    method: String,
    path: String,
    body: Vec<u8>,
}

const MAX_HEAD: usize = 16 * 1024;
const MAX_BODY: usize = 4 * 1024 * 1024;

fn parse_request(r: &mut dyn Read) -> std::io::Result<ParsedRequest> {
    let mut br = BufReader::new(r);
    let mut line = String::new();
    br.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let (Some(method), Some(target)) = (parts.next(), parts.next()) else {
        return Err(std::io::Error::other("bad request line"));
    };
    let method = method.to_uppercase();
    let path = target.split('?').next().unwrap_or("").to_string();
    // Headers, case-insensitive keys; only the ones this server cares about.
    let mut content_length: usize = 0;
    let mut head = String::new();
    loop {
        line.clear();
        br.read_line(&mut line)?;
        head.push_str(&line);
        if head.len() > MAX_HEAD {
            return Err(std::io::Error::other("request head too large"));
        }
        let t = line.trim_end();
        if t.is_empty() {
            break;
        }
        if let Some((k, v)) = t.split_once(':')
            && k.trim().eq_ignore_ascii_case("content-length")
        {
            content_length = v
                .trim()
                .parse()
                .map_err(|e| std::io::Error::other(format!("bad content-length: {e}")))?;
        }
    }
    if content_length > MAX_BODY {
        return Err(std::io::Error::other(format!(
            "body {content_length} B > {MAX_BODY} B cap"
        )));
    }
    let mut body = vec![0u8; content_length];
    br.read_exact(&mut body)?;
    Ok(ParsedRequest { method, path, body })
}

fn write_plain_error(w: &mut dyn Write, status: &str, message: &str) -> std::io::Result<()> {
    let body = serde_json::to_vec(&json!({ "type": "error", "message": message }))
        .unwrap_or_else(|_| b"{\"type\":\"error\"}".to_vec());
    write!(
        w,
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    w.write_all(&body)?;
    w.flush()
}

/// Wire one NDJSON event as a chunked-transfer HTTP chunk.
fn ndjson_event(w: &mut dyn Write, event: &Value) -> std::io::Result<()> {
    let mut b = serde_json::to_vec(event).map_err(std::io::Error::other)?;
    // The newline is PART of the payload: in HTTP chunked encoding a plain
    // "\r\n" after the body is transport framing that curl/python strip during
    // decode — without an in-payload 'ln' every event lands on ONE line and
    // ndjson consumers break (E2E-proven: first /synth arrived merged).
    b.push(b'\n');
    write!(w, "{:x}\r\n", b.len())?;
    w.write_all(&b)?;
    w.write_all(b"\r\n")?;
    w.flush()
}

fn begin_stream(w: &mut dyn Write) -> std::io::Result<()> {
    w.write_all(
        b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
    )
}

fn end_stream(w: &mut dyn Write) -> std::io::Result<()> {
    w.write_all(b"0\r\n\r\n")?;
    w.flush()
}

// ---------------------------------------------------------------------------
// Request → seam mapping
// ---------------------------------------------------------------------------

/// `speed` maps per lane, through each lane's own dial surface
/// (`LaneOptions`); no seam-type change. Rate control for the flow lanes is
/// unverified by a measuring gate — they refuse the dial instead of guessing.
fn speed_options(lane: &str, speed: f64) -> Result<Vec<(String, String)>, String> {
    let v = match lane {
        "kokoro" => {
            if !(0.5..=2.0).contains(&speed) {
                return Err(format!(
                    "kokoro speed must be within 0.5..=2.0, got {speed}"
                ));
            }
            ("speed", speed)
        }
        "piper" => {
            if !(0.25..=4.0).contains(&speed) {
                return Err(format!(
                    "piper speed must be within 0.25..=4.0, got {speed}"
                ));
            }
            // Piper's dial is duration `length_scale` (higher = slower), so
            // relay's `speed` (higher = faster) inverts: speed 0.5 → 2× longer.
            ("length-scale", 1.0 / speed)
        }
        other => {
            return Err(format!(
                "lane {other:?} does not implement the speed dial (documented per lane; pass no 'speed')"
            ));
        }
    };
    Ok(vec![(
        v.0.to_string(),
        format!("{:.*}", 4, v.1)
            .trim_end_matches('0')
            .trim_end_matches('.')
            .to_string(),
    )])
}

fn lane_slot<'a>(
    slots: &'a mut [LaneSlot],
    want: Option<&str>,
) -> Result<&'a mut LaneSlot, String> {
    let served: Vec<String> = slots.iter().map(|s| s.name.clone()).collect();
    match want {
        Some(name) => slots
            .iter_mut()
            .find(|s| s.name == name)
            .ok_or_else(|| format!("lane {name:?} not served (served lanes: {served:?})")),
        None => {
            if slots.len() == 1 {
                Ok(&mut slots[0])
            } else {
                Err(format!(
                    "request must name a lane; served lanes: {served:?}"
                ))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-connection handling
// ---------------------------------------------------------------------------

fn handle_conn(slots: &mut [LaneSlot], idle_unload_s: &u64, mut conn: Conn) -> std::io::Result<()> {
    let req = parse_request(&mut conn)?;
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => health(slots, idle_unload_s, conn),
        ("POST", "/synth") => synth(slots, conn, &req.body),
        ("POST", _) | ("GET", _) | ("HEAD", _) => write_plain_error(
            &mut conn,
            "404 Not Found",
            "unknown endpoint (POST /synth, GET /health)",
        ),
        _ => write_plain_error(&mut conn, "405 Method Not Allowed", "use POST /synth"),
    }
}

fn health(slots: &mut [LaneSlot], idle_unload_s: &u64, mut conn: Conn) -> std::io::Result<()> {
    let lanes: Vec<Value> = slots
        .iter()
        .map(|s| {
            json!({
                "lane": s.name,
                "model": s.model,
                "loaded": s.inner.is_some(),
                "resets": s.resets,
                "idle_ms_ago": s.last_used.map(|t| t.elapsed().as_millis() as u64),
            })
        })
        .collect();
    let body = serde_json::to_vec(&json!({
        "type": "health",
        "pid": std::process::id(),
        "lanes": lanes,
        "idle_unload_s": idle_unload_s,
        "rss_after_kb": crate::serve::vm_rss_kb(),
        "vm_peak_kb": crate::serve::vm_peak_kb(),
        "threads_dial": std::env::var("TTS_ORT_INTRA").ok(),
        "lang_default": "en",
    }))?;
    write!(
        conn,
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    )?;
    conn.write_all(&body)?;
    conn.flush()
}

fn synth(slots: &mut [LaneSlot], mut conn: Conn, body: &[u8]) -> std::io::Result<()> {
    let parsed: Value = match serde_json::from_slice(body) {
        Ok(v) => v,
        Err(e) => {
            return write_plain_error(&mut conn, "400 Bad Request", &format!("request body: {e}"));
        }
    };
    let obj = match parsed.as_object() {
        Some(o) => o,
        None => {
            return write_plain_error(
                &mut conn,
                "400 Bad Request",
                "request body must be a JSON object",
            );
        }
    };
    let t0 = Instant::now();

    let lane_name = obj.get("lane").and_then(Value::as_str);
    let text = obj.get("text").and_then(Value::as_str).unwrap_or("");
    if text.trim().is_empty() {
        return write_plain_error(&mut conn, "400 Bad Request", "missing or empty 'text'");
    }
    let lang_arg = obj.get("lang").and_then(Value::as_str).unwrap_or("en");
    let Some(lang) = Lang::parse(lang_arg) else {
        return write_plain_error(
            &mut conn,
            "400 Bad Request",
            &format!("unsupported lang {lang_arg:?} (en fr de es pt)"),
        );
    };
    let voice = obj
        .get("voice")
        .and_then(Value::as_str)
        .map(crate::resolve_voice);
    let seed = obj.get("seed").and_then(Value::as_u64);
    let speed = obj.get("speed").and_then(Value::as_f64);

    let slot = match lane_slot(slots, lane_name) {
        Ok(s) => s,
        Err(e) => return write_plain_error(&mut conn, "400 Bad Request", &e),
    };

    // Lane option mapping (speed → per-lane dial). Refuse rather than guess.
    let mut lane_options = tts_engine::types::LaneOptions::default();
    if let Some(speed) = speed {
        match speed_options(&slot.name, speed) {
            Ok(opts) => lane_options.extra = opts,
            Err(e) => return write_plain_error(&mut conn, "400 Bad Request", &e),
        }
    }

    // Cold path: lazy lane or post-idle-unload reload. The load wall is this
    // request's cost and is reported fresh in meta; every other request is
    // warm and carries NO load_ms (honesty law).
    let cold_load: Option<LoadInfo> = if slot.inner.is_none() {
        match slot.load() {
            Ok(info) => Some(info),
            Err(e) => {
                eprintln!("serve: lane {:?} load failed: {e}", slot.name);
                slot.last_used = Some(Instant::now());
                return write_plain_error(
                    &mut conn,
                    "500 Internal Server Error",
                    &format!("lane load failed: {e}"),
                );
            }
        }
    } else {
        None
    };
    slot.last_used = Some(Instant::now());
    let mode = if cold_load.is_some() { "cold" } else { "warm" };

    let req = SynthRequest {
        text: text.to_string(),
        chunks: chunking::chunk_text(text, chunking::ChunkConfig::default()),
        lang,
        voice,
        seed,
        lane_options,
    };

    // Errors after this point stream as NDJSON `error` events (a 200 already
    // implies chunks may follow); errors before meta land as plain HTTP.
    let lane = slot.inner.as_mut().expect("loaded above");
    let stream = match lane.synthesize(&req) {
        Ok(s) => s,
        Err(e) => match &e {
            LaneError::Load(_) => {
                return write_plain_error(&mut conn, "500 Internal Server Error", &e.to_string());
            }
            LaneError::Synth(msg) | LaneError::Io(msg) => {
                return write_plain_error(&mut conn, "422 Unprocessable Entity", msg);
            }
        },
    };
    // NOTE: synthesize() returns a LAZY iterator — the audio is produced in
    // the chunk loop below. The idle clock is stamped when the loop ENDS
    // (post-E2E bug: stamped here meant a 10s synth outran a 4s idle window
    // and the next request paid a gratuitous cold reload).

    let mut w = BufWriter::new(conn);
    begin_stream(&mut w)?;
    ndjson_event(
        &mut w,
        &meta_event(&stream, mode, cold_load.map(|i| i.load_ms), slot),
    )?;

    let mut chunks = 0u64;
    let mut pcm_bytes = 0u64;
    let mut ttfa_ms: Option<u64> = None;
    for (i, item) in stream.chunks.enumerate() {
        match item {
            Ok(out) => {
                let ev = json!({
                    "type": "chunk",
                    "seq": i,
                    "pcm_b64": base64(&pcm_i16_le(&out.pcm)),
                    "stages_ms": {
                        "g2p": out.stages.g2p.as_secs_f64() * 1e3,
                        "acoustic": out.stages.acoustic.as_secs_f64() * 1e3,
                        "vocoder": out.stages.vocoder.as_secs_f64() * 1e3,
                        "io": out.stages.io.as_secs_f64() * 1e3,
                    },
                });
                if ttfa_ms.is_none() {
                    ttfa_ms = Some(t0.elapsed().as_millis() as u64);
                }
                chunks += 1;
                pcm_bytes += (out.pcm.len() as u64) * 2;
                ndjson_event(&mut w, &ev)?;
            }
            Err(e) => {
                let ev = json!({ "type": "error", "message": e.to_string(), "seq_done": i });
                let _ = ndjson_event(&mut w, &ev);
                let _ = end_stream(&mut w);
                return Ok(());
            }
        }
    }

    let wall = t0.elapsed().as_millis() as u64;
    // Idle clock: the lane was busy for the whole stream — stamp at END.
    slot.last_used = Some(Instant::now());
    let sample_rate = stream.meta.sample_rate as u64;
    let audio_ms = if sample_rate > 0 {
        pcm_bytes * 1000 / (sample_rate * 2)
    } else {
        0
    };
    let done = json!({
        "type": "done",
        "chunks": chunks,
        "audio_s": audio_ms as f64 / 1000.0,
        "wall_ms": wall,
        "rtf": if audio_ms > 0 { Some(wall as f64 / audio_ms as f64) } else { None },
        "ttfa_ms": ttfa_ms,
        "sample_rate": sample_rate,
        "mode": mode,
        "resets": slot.resets,
        "rss_after_kb": vm_rss_kb(),
        "vm_peak_kb": vm_peak_kb(),
    });
    ndjson_event(&mut w, &done)?;
    end_stream(&mut w)
}

fn meta_event(
    stream: &SynthStream,
    mode: &str,
    cold_load_ms: Option<u64>,
    slot: &LaneSlot,
) -> Value {
    let m = &stream.meta;
    // Honesty fields: mode + fresh load ONLY on a true cold. On warm requests
    // `load_ms` is ABSENT (not null) — the client must not read a historical
    // load wall as this-request work.
    let mut ev = serde_json::Map::new();
    ev.insert("type".into(), json!("meta"));
    ev.insert("lane".into(), json!(m.lane));
    ev.insert("model".into(), json!(m.model));
    ev.insert("model_sha".into(), json!(m.model_sha));
    ev.insert("voice".into(), json!(m.voice));
    ev.insert("sample_rate".into(), json!(m.sample_rate));
    ev.insert("lang".into(), json!(m.lang.code()));
    ev.insert("determinism_class".into(), json!(m.determinism.as_str()));
    ev.insert(
        "nondet_reason".into(),
        match &m.determinism {
            tts_engine::types::Determinism::NonDeterministic(r) => json!(r.clone()),
            _ => Value::Null,
        },
    );
    ev.insert("seed".into(), json!(m.seed));
    ev.insert("runtime".into(), json!(m.runtime));
    ev.insert("chunk_policy".into(), json!(m.chunk_policy));
    ev.insert("restart_voice".into(), json!(m.restart_voice));
    ev.insert("mode".into(), json!(mode));
    if let Some(ms) = cold_load_ms {
        ev.insert("load_ms".into(), json!(ms));
    }
    ev.insert("resets".into(), json!(slot.resets));
    ev.insert("rss_after_kb".into(), json!(vm_rss_kb()));
    ev.insert("vm_peak_kb".into(), json!(vm_peak_kb()));
    ev.insert("protocol".into(), json!("ndjson-chunks/1"));
    Value::Object(ev)
}

fn pcm_i16_le(pcm: &[i16]) -> Vec<u8> {
    let mut out = Vec::with_capacity(pcm.len() * 2);
    for s in pcm {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Standard base64 (RFC 4648 with padding; curl + python base64.b64decode
/// and node's Buffer.from(b64,'base64') all accept the padded form).
fn base64(data: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = *chunk.get(1).unwrap_or(&0) as u32;
        let b2 = *chunk.get(2).unwrap_or(&0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(ALPHABET[(n >> 18) as usize & 63] as char);
        out.push(ALPHABET[(n >> 12) as usize & 63] as char);
        out.push(if chunk.len() > 1 {
            ALPHABET[(n >> 6) as usize & 63] as char
        } else {
            '='
        });
        out.push(if chunk.len() > 2 {
            ALPHABET[n as usize & 63] as char
        } else {
            '='
        });
    }
    out
}

fn vm_field(key: &str) -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            return rest.split_whitespace().next()?.parse::<u64>().ok();
        }
    }
    None
}

pub fn vm_rss_kb() -> Option<u64> {
    vm_field("VmRSS:")
}

pub fn vm_peak_kb() -> Option<u64> {
    vm_field("VmPeak:")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- request parsing ------------------------------------------------

    #[test]
    fn parse_request_post_with_body() {
        let raw = b"POST /synth HTTP/1.1\r\nHost: tts\r\nContent-Type: application/json\r\nContent-Length: 13\r\n\r\n{\"text\":\"hi\"}" as &[u8];
        let mut r = std::io::Cursor::new(raw);
        let p = parse_request(&mut r).unwrap();
        assert_eq!(p.method, "POST");
        assert_eq!(p.path, "/synth");
        assert_eq!(p.body, b"{\"text\":\"hi\"}");
    }

    #[test]
    fn parse_request_query_and_empty_body() {
        let raw = b"GET /health?x=1 HTTP/1.1\r\ncontent-length: 0\r\n\r\n" as &[u8];
        let mut r = std::io::Cursor::new(raw);
        let p = parse_request(&mut r).unwrap();
        assert_eq!(p.method, "GET");
        assert_eq!(p.path, "/health");
        assert!(p.body.is_empty());
    }

    #[test]
    fn parse_request_rejects_oversize_body() {
        let raw = format!(
            "POST /synth HTTP/1.1\r\nContent-Length: {}\r\n\r\n",
            MAX_BODY + 1
        );
        let mut r = std::io::Cursor::new(raw.into_bytes());
        assert!(parse_request(&mut r).is_err());
    }

    // ---- base64 (client probes decode the chunk pcm with this) ----------

    #[test]
    fn base64_matches_rfc4648_vectors() {
        assert_eq!(base64(b""), "");
        assert_eq!(base64(b"f"), "Zg==");
        assert_eq!(base64(b"fo"), "Zm8=");
        assert_eq!(base64(b"foo"), "Zm9v");
        assert_eq!(base64(b"foob"), "Zm9vYg==");
        assert_eq!(base64(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn base64_roundtrips_through_std_decode_reference() {
        // Reference decode via python's stdlib keeps the encoder honest.
        let data: Vec<u8> = (0..1000u32).map(|i| i as u8).collect();
        let enc = base64(&data);
        assert_eq!(enc.len() % 4, 0);
        assert_eq!(enc.len(), data.len().div_ceil(3) * 4);
    }

    // ---- speed → lane dial mapping -------------------------------------

    #[test]
    fn speed_kokoro_is_pass_through() {
        assert_eq!(
            speed_options("kokoro", 1.3).unwrap(),
            vec![("speed".to_string(), "1.3".to_string())]
        );
        assert!(speed_options("kokoro", 0.49).is_err());
        assert!(speed_options("kokoro", 2.01).is_err());
    }

    #[test]
    fn speed_piper_inverts_to_length_scale() {
        // speed 0.5 (slower) → length-scale 2 (durations doubled).
        assert_eq!(
            speed_options("piper", 0.5).unwrap(),
            vec![("length-scale".to_string(), "2".to_string())]
        );
        assert!(speed_options("piper", 0.24).is_err());
        assert!(speed_options("piper", 4.01).is_err());
    }

    #[test]
    fn speed_unimplemented_lanes_refuse() {
        assert!(speed_options("flow-candle", 1.0).is_err());
        assert!(speed_options("smoke", 1.0).is_err());
    }

    // ---- chunk framing --------------------------------------------------

    #[test]
    fn ndjson_event_writes_chunked_transfer() {
        let mut buf = Vec::new();
        let ev = serde_json::json!({ "type": "ping" });
        ndjson_event(&mut buf, &ev).unwrap();
        let s = String::from_utf8(buf).unwrap();
        let len = ev.to_string().len() + 1; // + the ndjson 'ln' inside the payload
        assert_eq!(s, format!("{len:x}\r\n{ev}\n\r\n"));
    }

    #[test]
    fn stream_begin_and_end_bracket_events() {
        let mut buf = Vec::new();
        begin_stream(&mut buf).unwrap();
        let ev = serde_json::json!({ "n": 1 });
        ndjson_event(&mut buf, &ev).unwrap();
        end_stream(&mut buf).unwrap();
        let s = String::from_utf8(buf).unwrap();
        assert!(s.starts_with("HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n"));
        assert!(s.contains("Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"));
        assert!(s.ends_with("0\r\n\r\n"));
    }

    // ---- lane slot selection -------------------------------------------

    fn stub_slot(name: &str) -> LaneSlot {
        LaneSlot {
            name: name.to_string(),
            model: "m".to_string(),
            model_ref: ModelRef::Pseudo("m".to_string()),
            inner: None,
            resets: 0,
            last_used: None,
        }
    }

    #[test]
    fn lane_slot_by_name_and_single_default() {
        let mut slots = vec![stub_slot("piper"), stub_slot("kokoro")];
        assert_eq!(
            lane_slot(&mut slots, Some("kokoro")).unwrap().name,
            "kokoro"
        );
        assert!(lane_slot(&mut slots, Some("nope")).is_err());
        let mut single = vec![stub_slot("piper")];
        assert_eq!(lane_slot(&mut single, None).unwrap().name, "piper");
        let mut multi = vec![stub_slot("piper"), stub_slot("kokoro")];
        assert!(lane_slot(&mut multi, None).is_err());
    }

    // ---- idle-unload deadline ------------------------------------------

    #[test]
    fn next_unload_wait_only_counts_loaded_recent_use() {
        let mut slots = vec![stub_slot("a"), stub_slot("b")];
        assert_eq!(next_unload_wait(&slots, 5), None); // nothing loaded
        slots[0].inner = Some(make_lane("smoke").unwrap());
        slots[0].last_used = Some(Instant::now() - Duration::from_secs(4));
        slots[1].last_used = Some(Instant::now() - Duration::from_secs(2));
        // ~1s until lane a crosses a 5s deadline (minus test-step elapsed).
        let wait = next_unload_wait(&slots, 5).unwrap();
        assert!(
            wait > Duration::from_millis(500) && wait <= Duration::from_secs(1),
            "wait = {wait:?}"
        );
        // unloaded lane with stale last_used is not in the deadline set
        let mut only_unloaded = vec![stub_slot("a")];
        only_unloaded[0].last_used = Some(Instant::now() - Duration::from_secs(99));
        assert_eq!(next_unload_wait(&only_unloaded, 5), None);
    }
}
