//! The `naz` binary's `--rescale-q` is the search the Lambda engine runs for a standard game by
//! default (`RESCALE_Q=standard`: `Agent::with_rescale_q_modes(&[Mode::Standard])`), so
//! `naz --seed <tie_seed> --starting N --rescale-q` reproduces a live move and its root
//! statistics, and without the flag naz is the plain (Python) search.
//!
//! The position is a 4-snake standard game with two snakes left (a 1v1 endgame), on which the
//! rescaled and the plain 256-simulation search choose different moves. Needs the run dir
//! (`$NAZ_RUN`, default the deployed M network); without it the test prints SKIP.

use serde_json::{Value, json};
use slinky_rules::Mode;
use slinky_search::{Agent, AgentMove};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// `tie_seed("example-standard", "8")` of the Lambda engine.
const SEED: u32 = 1_491_447_951;

fn run_dir() -> Option<PathBuf> {
    let d = PathBuf::from(
        std::env::var("NAZ_RUN")
            .unwrap_or_else(|_| "/home/nathan/slinky-serve/az-modes-M-s0".into()),
    );
    if d.join("params.npz").exists() {
        Some(d)
    } else {
        eprintln!("SKIP: no run dir {}", d.display());
        None
    }
}

fn examples() -> Value {
    serde_json::from_str(include_str!("../../server/data/examples.json")).unwrap()
}

/// The example standard game at turn 8 with its second and third snakes left.
fn endgame() -> Value {
    let mut r = examples()["requests"]["standard"].clone();
    r["turn"] = json!(8);
    let snakes = r["board"]["snakes"].as_array().unwrap()[1..3].to_vec();
    r["you"] = snakes[0].clone();
    r["board"]["snakes"] = json!(snakes);
    r
}

/// Runs `naz -v` and returns the move it printed and its per-seat lines.
fn naz(run: &Path, args: &[&str], request: &Value) -> (String, Vec<String>, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_naz"))
        .arg("--run")
        .arg(run)
        .args(args)
        .arg("-v")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(request.to_string().as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "naz {args:?}: {:?}", out.status);
    let stdout: Value = serde_json::from_slice(&out.stdout).unwrap();
    let stderr = String::from_utf8(out.stderr).unwrap();
    let seats = stderr
        .lines()
        .filter(|l| l.starts_with("seat "))
        .map(str::to_string)
        .collect();
    (stdout["move"].as_str().unwrap().to_string(), seats, stderr)
}

/// The per-seat lines `naz -v` prints for an agent's move.
fn seat_lines(m: &AgentMove) -> Vec<String> {
    let s = &m.search;
    (0..m.num_snakes)
        .map(|p| {
            format!(
                "seat {p}: move {} visits {:?} q {:?} prior {:?}",
                s.action[p], s.visits[p], s.q[p], s.root_prior[p]
            )
        })
        .collect()
}

#[test]
fn rescale_q_flag_is_the_engines_standard_search() {
    let Some(run) = run_dir() else { return };
    let request = endgame();
    let seed = SEED.to_string();
    let args = ["--seed", &seed, "--starting", "4"];
    let body = request.to_string();

    // What the engine builds for RESCALE_Q=standard, and the plain agent.
    let engine = Agent::load(&run)
        .unwrap()
        .with_rescale_q_modes(&[Mode::Standard])
        .play(&body, Some(4), SEED)
        .unwrap();
    let plain = Agent::load(&run)
        .unwrap()
        .play(&body, Some(4), SEED)
        .unwrap();
    assert_ne!(engine.search.visits, plain.search.visits);
    assert_ne!(
        engine.mv, plain.mv,
        "the position no longer separates the rescaled and the plain search"
    );

    let (mv, seats, stderr) = naz(&run, &[&args[..], &["--rescale-q"]].concat(), &request);
    assert!(stderr.contains("rescale_q on"), "{stderr}");
    assert_eq!(mv, engine.mv.name());
    assert_eq!(seats, seat_lines(&engine));

    let (mv, seats, stderr) = naz(&run, &args, &request);
    assert!(stderr.contains("rescale_q off"), "{stderr}");
    assert_eq!(mv, plain.mv.name());
    assert_eq!(seats, seat_lines(&plain));
    eprintln!(
        "naz --rescale-q plays {}, plain naz {}; visits of seat 0: {:?} against {:?}",
        engine.mv.name(),
        plain.mv.name(),
        engine.search.visits[0],
        plain.search.visits[0]
    );
}

/// Only standard games are rescaled: the flag changes nothing in the other modes, as the
/// engine's default `RESCALE_Q=standard` does not.
#[test]
fn rescale_q_flag_leaves_other_modes_alone() {
    let Some(run) = run_dir() else { return };
    let ex = examples();
    for mode in ["duel", "royale", "constrictor"] {
        let req = &ex["requests"][mode];
        let with = naz(&run, &["--rescale-q"], req);
        let without = naz(&run, &[], req);
        assert_eq!((with.0, with.1), (without.0, without.1), "{mode}");
    }
}
