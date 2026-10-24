//! `deploy/lambda-native/deploy.sh` and the engine take the same `RESCALE_Q` values.
//!
//! `deploy.sh` checks `RESCALE_Q` before it updates the live function, and the engine
//! (`Settings::from_env`, `parse_modes`) fails at init on a value it rejects, which answers
//! nothing from every new container until a redeploy. So the check must accept what the engine
//! accepts and reject what it rejects. This runs the Python block of `deploy.sh` (the one that
//! merges and checks the function's environment) on every value of `tests/rescale_q_cases.json`;
//! the engine's side of the same table is the unit test `rescale_q_case_table` in `engine.rs`.
//!
//! Needs `python3` and the repository's `deploy/` directory; without them the test prints SKIP.

use serde_json::Value;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

/// The Python program of the `python3 - "$env_file" <<'PY'` heredoc of `deploy.sh`.
fn python_block() -> Option<String> {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../deploy/lambda-native/deploy.sh");
    let Ok(script) = std::fs::read_to_string(&path) else {
        eprintln!("SKIP: no {}", path.display());
        return None;
    };
    let mut lines = script.lines();
    assert!(
        lines
            .find(|l| l.starts_with("DEPLOYED=") && l.contains("<<'PY'"))
            .is_some(),
        "deploy.sh has no `DEPLOYED=... python3 - ... <<'PY'` block"
    );
    let block: Vec<&str> = lines.take_while(|l| *l != "PY").collect();
    assert!(
        block.iter().any(|l| l.contains("RESCALE_Q")),
        "the python block of deploy.sh does not mention RESCALE_Q"
    );
    Some(block.join("\n") + "\n")
}

/// Runs `block` as `deploy.sh` does, with `RESCALE_Q` set to `value` (or unset), and no deployed
/// environment. Returns whether it exited successfully, its stderr and the variables it wrote.
fn run(block: &str, value: Option<&str>, tag: &str) -> Option<(bool, String, Value)> {
    let out = std::env::temp_dir().join(format!("deploy-sh-{}-{tag}.json", std::process::id()));
    let mut cmd = Command::new("python3");
    cmd.args(["-I", "-", out.to_str().unwrap()])
        .env_clear()
        .env("PATH", "/usr/bin:/bin")
        .env("DEPLOYED", "")
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    if let Some(v) = value {
        cmd.env("RESCALE_Q", v);
    }
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("SKIP: no python3 ({e})");
            return None;
        }
    };
    child
        .stdin
        .take()
        .unwrap()
        .write_all(block.as_bytes())
        .unwrap();
    let o = child.wait_with_output().unwrap();
    let written = std::fs::read_to_string(&out)
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .map_or(Value::Null, |v| v["Variables"].clone());
    let _ = std::fs::remove_file(&out);
    Some((
        o.status.success(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
        written,
    ))
}

#[test]
fn deploy_sh_checks_rescale_q_like_the_engine() {
    let Some(block) = python_block() else { return };
    let cases: Value = serde_json::from_str(include_str!("rescale_q_cases.json")).unwrap();
    let cases = cases.as_array().unwrap();
    assert!(cases.len() >= 40);
    let (mut accepted, mut rejected) = (0, 0);
    for (i, c) in cases.iter().enumerate() {
        let value = c["value"].as_str().unwrap();
        // The engine accepts the value when the table lists the modes it gives.
        let engine_accepts = c["modes"].is_array();
        let Some((ok, stderr, vars)) = run(&block, Some(value), &i.to_string()) else {
            return;
        };
        assert_eq!(
            ok,
            engine_accepts,
            "RESCALE_Q={value:?}: the engine {} it, deploy.sh {} it ({stderr})",
            if engine_accepts { "accepts" } else { "rejects" },
            if ok { "accepts" } else { "rejects" },
        );
        if ok {
            accepted += 1;
            // Passed on as given: the engine does the parsing.
            assert_eq!(vars["RESCALE_Q"], value, "RESCALE_Q={value:?}");
        } else {
            rejected += 1;
            assert!(
                stderr.contains("RESCALE_Q"),
                "RESCALE_Q={value:?}: {stderr}"
            );
        }
    }
    eprintln!(
        "deploy.sh vs engine on {} RESCALE_Q values: {accepted} accepted, {rejected} rejected, all agree",
        cases.len()
    );
    assert!(accepted > 10 && rejected > 10);
    // Not set: nothing is checked and nothing is written (the engine's default, or the deployed value).
    let (ok, stderr, vars) = run(&block, None, "unset").unwrap();
    assert!(ok, "{stderr}");
    assert!(vars.get("RESCALE_Q").is_none());
}
