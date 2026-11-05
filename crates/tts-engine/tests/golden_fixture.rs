//! Integration test: the stored golden fixture (docs/seam.md §9) must satisfy
//! every §5.1 parser invariant. This is the contract T003's parser re-asserts
//! on live runs.

use serde_json::Value;

fn fixture() -> Vec<Value> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../eval/fixtures/golden-chunks.jsonl"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("bad row {l:?}: {e}")))
        .collect()
}

#[test]
fn golden_fixture_satisfies_seam_invariants() {
    let rows = fixture();
    assert!(rows.len() >= 3, "header + ≥1 chunk + summary");
    assert_eq!(rows[0]["type"], "header");
    assert_eq!(rows.last().unwrap()["type"], "summary");
    let header = &rows[0];
    for f in [
        "lane",
        "model",
        "voice",
        "sample_rate",
        "lang",
        "determinism",
        "seed",
        "lane_version",
        "runtime",
        "load_ms",
        "chunk_policy",
        "restart_voice",
    ] {
        assert!(header.get(f).is_some(), "header missing {f}");
    }

    let chunks = &rows[1..rows.len() - 1];
    // i monotonic from 0
    for (k, row) in chunks.iter().enumerate() {
        assert_eq!(row["type"], "chunk");
        assert_eq!(row["i"], k as u64);
        for f in [
            "t_synth_start_ms",
            "chunk_ms",
            "ttfa_ms",
            "rtf_running",
            "peak_rss_kb",
            "pcm_bytes",
        ] {
            assert!(row.get(f).is_some(), "chunk missing {f}");
        }
        for f in ["g2p", "acoustic", "vocoder", "io"] {
            assert!(row["chunk_ms"].get(f).is_some(), "chunk_ms missing {f}");
        }
        // t_synth_start_ms constant across the run (synthesize() call instant)
        assert_eq!(row["t_synth_start_ms"], chunks[0]["t_synth_start_ms"]);
    }

    // ttfa on first chunk only
    let ttfa_rows: Vec<&Value> = chunks.iter().filter(|r| !r["ttfa_ms"].is_null()).collect();
    assert_eq!(ttfa_rows.len(), 1, "ttfa set exactly once");
    assert_eq!(ttfa_rows[0]["i"], chunks[0]["i"]);

    // rtf_running: finite positive cumulative ratios (NOT contractually monotone)
    for r in chunks {
        let rtf = r["rtf_running"].as_f64().expect("rtf_running f64");
        assert!(rtf.is_finite() && rtf > 0.0);
    }

    // summary consistency
    let summary = rows.last().unwrap();
    assert_eq!(summary["chunks"], chunks.len() as u64);
    let rtf_final = summary["rtf_final"].as_f64().unwrap();
    assert_eq!(
        rtf_final,
        chunks.last().unwrap()["rtf_running"].as_f64().unwrap(),
        "rtf_final == last rtf_running"
    );
    assert_eq!(summary["ttfa_ms"], ttfa_rows[0]["ttfa_ms"]);
    // gate enum
    let gate = summary["gate"]["byte_identity"].as_str().unwrap();
    assert!(["pass", "fail", "not-assessed"].contains(&gate));
    for r in chunks {
        assert!(r["pcm_bytes"].as_str().unwrap().starts_with("sha256:"));
    }
    // stage sums fit under per-chunk flush deltas (fixture-specific)
    let sums: Vec<u64> = chunks
        .iter()
        .map(|r| {
            ["g2p", "acoustic", "vocoder", "io"]
                .iter()
                .map(|f| r["chunk_ms"][f].as_u64().unwrap())
                .sum()
        })
        .collect();
    assert!(sums[0] <= ttfa_rows[0]["ttfa_ms"].as_u64().unwrap());
}
