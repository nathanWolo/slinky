# slinky-lambda

The native AlphaZero snakes on AWS Lambda, two ways:

* **Deployed: `slinky-engine` inside the Python Lambda** (`deploy/lambda/`, function
  `slinky-snake`). The Python handler serves the numpy snakes and, for the native ones
  (`lambda_function.NATIVE`: `/az-256`), starts `bin/slinky-engine` once per container and
  talks to it in JSON lines (`src/bin/engine.rs`: one request line in, one answer line out,
  logs on stderr). One function, one URL and one bill for every snake. See
  `deploy/lambda/README.md`.
* **Deployed: a standalone custom-runtime handler** (`provided.al2023`, `src/main.rs`, binary
  `slinky-lambda`), described below: function `slinky-native` (`/az-256` and the 200 ms
  `/az-t200`), see `deploy/lambda-native/README.md`. Its containers start in tens of ms, where
  the Python function's take ~410 ms.

Both use the same handler (`App`) and engine (`SnakeEngine`). The standalone one routes like
the Python deployment behind one Function URL:

    GET  /<snake>          info (apiversion, author, color, head, tail, version)
    POST /<snake>/start    remembers the game's snake count (bounded per-process cache)
    POST /<snake>/move     {"move": "up|down|left|right"}: the az_modes search, safe move on any error
    POST /<snake>/end      logged

The engine is `slinky_search::Agent`: the weights are read once per container (at init, 1.6 ms for
the M network), each move is the decoupled-PUCT search of the Python agent
(`az_modes:<run>:simulations=256`) with the standard-game rescale of `RESCALE_Q`, which is the
Python search exactly only with `RESCALE_Q=none` (see Rescaled values), unless the settings below
ask for more. Layout: `src/engine.rs`
(settings, seed policy, search and fallback), `src/lib.rs` (routing, `/start` cache, logs),
`src/main.rs` (Lambda glue), `tests/handler.rs`.

## Environment variables

| variable | meaning | default |
|---|---|---|
| `WEIGHTS_DIR` | root of the weights; a snake's run dir (`config.json`, `params.npz`) is `WEIGHTS_DIR/<subdir>` | `/var/task/weights` |
| `SNAKES` | `name=subdir[:simulations=N][:budget_ms=T],...`: the URL name of each snake and its weights; the per-snake options override the global ones | `slinky=az-modes` |
| `SIMULATIONS` | simulations per move; with `BUDGET_MS` it is the cap | 256 (16384 when only `BUDGET_MS` is set) |
| `BUDGET_MS` | stop searching this many ms after the request reached the handler (not the faithful mode: the simulation count depends on the CPU) | off |
| `THREADS` | root-parallel search trees, one per vCPU (`1` is the plain search) | 1 |
| `TIE_SEED` | one fixed tie-break seed for every move (tests) | derived, see below |
| `MEMORY_LIMIT_MB` | cap on one search tree (the search stops early when it is reached). With `REUSE` a snake holds up to 8 trees (2 x `GAME_TREES`: the current and the previous tree of each of 4 games), see Memory | 256 |
| `EARLY_STOP` | `0`: always run the whole simulation count; `1`: stop once the leading move cannot be overtaken by the simulations left (the move is the same). A `BUDGET_MS` snake stops early only with `REUSE=0`, see Early stop and reuse | `1` |
| `REUSE` | `0`: forget each search; `1`: keep each game's last tree (the 4 most recent games) and reuse its network evaluations (bit-identical moves) | `1` |
| `RESCALE_Q` | modes (comma list of `standard`, `duel`, `royale`, `constrictor`, or `none`) whose search rescales each snake's values to the range it can still reach; for every snake; changes moves, see Rescaled values | `standard` |
| `DEADLINE_MARGIN_MS` | time kept for the network and Lambda's overhead: every search stops at `game.timeout` less this (see Deadline) | 150 |
| `DEBUG_PANIC` | `search:<game id>` or `handler:<game id>`: panic on that game's `/move` (tests only) | off |
| `SNAKE_AUTHOR`, `SNAKE_HEAD`, `SNAKE_TAIL`, `SNAKE_COLOR_<NAME>` | info response (`<NAME>` upper case, `-` and `.` become `_`) | `slinky`, `default`, `default`, `#8b5cf6` |

A malformed setting (`SIMULATIONS=0`, a bad `SNAKES` item, two snakes with one name) makes the
function fail at init with a message, which shows in the first invocation's error. A snake whose
weights cannot be loaded keeps answering with the safe move and logs why (event `load`). Each snake's
`load` line also says what is live for it: `early_stop` and `reuse` (true or false, after `THREADS`
and `BUDGET_MS` are taken into account, see Early stop and reuse) and `rescale_q`, the list of modes
searched with rescaled values (`[]` for `none`).

### Deadline

Every search stops at the request's deadline, whatever its settings: `game.timeout` (500 ms
when the request has none, or one that is not a positive number) less `DEADLINE_MARGIN_MS`,
counted from the request's arrival at the handler, and never after the invocation's own
deadline less the margin. When the deadline cuts a search, the move is the best of the
simulations run so far (at least one always runs, unless the move is decided at once:
`StopReason::Decided`, 0 simulations on a forced move). `BUDGET_MS` past the deadline is cut to
it. A fixed `SIMULATIONS` search that ends before its deadline is unchanged (the 461 Python
goldens still agree, with `RESCALE_Q=none`). Each move's log line has `deadline_ms` (arrival to deadline) and `cut`
(a one-tree search was stopped by the deadline or the memory cap; a search that ended because
its move was decided or its tree solved is not cut).

### Early stop and reuse

Two speed-ups that never change a move; switch them off with `EARLY_STOP=0` and `REUSE=0`. With
the early stop a search ends once its move cannot change: for a `SIMULATIONS` snake (`/az-256`)
that makes `sims` below the count normal, and `sims` 0 on a forced move (one legal move). A
`BUDGET_MS` snake (`/az-t200`) counts to its cap (16,384 simulations), where the early stop could
only end forced moves, and a forced move's tree is the one the next move reuses most (47% of its
nodes in a duel and 29% in a 4-player game, against 33% and 16% after an ordinary move). So a
`BUDGET_MS` snake searches forced moves to its deadline when `REUSE` is on, and stops them at
once (`sims` 0) only with `REUSE=0`. With `REUSE` the engine keeps the last tree of each of the 4
most recent games, and the next move copies the network evaluation of every node it shares with
that tree (`reused` in the log line). Both apply to a one-tree search (`THREADS=1`).

Measured: on one x86 core a 256-simulation `/az-256` move takes 37-54% of the time it took before
the early stop, the reuse and the structured first layer, and a fresh 200 ms `/az-t200` search
reaches 20-23% more simulations, plus 4-43% from reuse. On Graviton2 (the `{"bench": ...}` report
of a Lambda test function, 1,769 MB, 2026-10-18), ms per 256-simulation move plain -> early stop
-> early stop + reuse: duel 23.2 -> 15.3 -> 12.6, standard 38.9 -> 29.5 -> 29.3, royale 39.2 ->
28.8 -> 27.7, constrictor 30.1 -> 14.2 -> 11.6, with the same move as the plain search in 240 of
240 positions. Simulations in a 200 ms search, fresh / with reuse: duel 2,341 / 3,019, standard
1,208 / 1,269, royale 2,402 / 3,428, constrictor 4,581 / 5,057.

### Rescaled values

In the modes of `RESCALE_Q` (default `standard`) the search maps each alive snake's values onto
the range it can still reach instead of the whole game's `[-1, 1]` (`SearchConfig::rescale_q`):
once two of four snakes are dead, both survivors have already outlasted two, so the 1v1
endgame's outcomes span `[1/3, 1]`, and on the fixed scale the PUCT priors outweigh those values
three times as much as in a duel.

Which moves change. The range comes from each tree node's own state, so a standard game of `n`
snakes is searched differently wherever its tree reaches a node with 1 to `n - 2` snakes dead,
also while all snakes are alive, and in 3-snake games. A node with nobody dead, one with a single
snake alive, and every node of a duel (`n <= 2`) keep the default map. Duel, royale and
constrictor are unchanged unless `RESCALE_Q` names them. The setting is global: `/az-256` and
`/az-t200` both get it, so `/az-256` is the Python agent's search only with `RESCALE_Q=none`.
On the 461 golden ladder requests (89 of them standard, 256 simulations, the golden seeds) it
changes the root visits of 37, 10 of them with every snake alive, and the move of 2.

`n` is the larger of the snake count `/start` left in this container's cache and the snakes
listed in the request. A container that missed the game's `/start` searches a 4-snake game with
2 left as a duel (no rescale) and one with 3 left with `n = 3`. The `{"bench"}` direct invocation
runs with the rescale off.

Evidence: on arena the snake converted 43% of its standard 1v1 endgames against 73-78% for the
top minimax snakes (2026-10-16 to 10-23). In self-play on the proxy weights at 50 ms a move,
rescaled against plain, 1000 games a mode: standard +26 Elo [+9, +44] (win share 36.9% against
a parity of 25%), royale -14 [-31, +4], constrictor +1 [-15, +17]; hence standard only. That test
did not use the deployed weights or budgets (256 simulations, 200 ms) and played 4-snake games;
a test on the deployed weights is in progress (2026-10-23).

### Memory

`MEMORY_LIMIT_MB` caps one search tree. With `REUSE` each of a snake's 4 kept games holds its
current and its previous tree, so a snake holds up to 8 trees (`2 x GAME_TREES`). At the
16,384-simulation cap of `/az-t200` that is an estimated worst case of about 360 MB (8 x the
45 MB resident measured for one such tree), which fits the 1,769 MB function. Lower the limit
if a snake is given a much larger cap or more time.

### Seed policy

The Python server draws the tie noise of its i-th move from `fold_in(key(0), i)` with a per-process
counter. A Lambda function has many containers and any of them may get a move, so a counter would
make the answer depend on the container. Instead the seed is a fixed hash of the request:
`tie_seed(game id, turn)` (FNV-1a 64 over the id, a separator and the turn, folded to 32 bits;
`engine::tie_seed`). The same request always gets the same move (a retry cannot flip a near-tie),
different games and turns draw different noise, and all containers agree. The noise is that of
`jax.random.key(seed)`, so `naz --seed <seed>` reproduces a Lambda move exactly (the seed is in
each move's log line; add `--starting N` for the snake count at `/start`, and `--rescale-q` for a
standard game, the default `RESCALE_Q`). `TIE_SEED` overrides it.

### Fallback

Any error answers HTTP 200 with the safe move (`slinky_api::safe_move`): a request the engine cannot
read, a game the network does not play (not 11x11, not 2 to 4 snakes, a mode the run was not
trained on: the Python heuristic fallback is not ported), missing weights. Bodies are read with
Python's `json.loads` dialect (`slinky_api::Request`: `NaN`, a BOM, lone surrogates, deep
nesting) and any `game.id` type is accepted, as the Python handler does; only bodies Python
refuses (not JSON, not an object, no `game.id`, over 64 KiB, base64 that does not decode) get a
400. The event is read as plain JSON with `lambda_runtime` (not `lambda_http`, whose event
conversion `expect`s valid base64 and a valid URI), so no request can panic outside the
handler. A panic in the handler or the search answers the safe move (a 500 for other routes)
and the process keeps serving, when the binary is built with `panic = "unwind"`: the
workspace default is `abort`, so the deploy builds set `CARGO_PROFILE_RELEASE_PANIC=unwind`
(`tests/runtime.rs` checks it with `DEBUG_PANIC` and prints SKIP for an abort build; run the
tests with that variable set to cover it). The weights' `params.npz` entries are checked
against their CRC-32 at load: a corrupted file is a load error (the snake answers the safe
move and logs why).

### `/start` cache limit

The snake count at `/start` is kept per container. The `/move` of a game may reach another
container (concurrent games make Lambda start more), which then plays with the living snakes
only, the case the Python server has after a restart (the network was trained for 3-seat variants
too, but a game seen that way is not the one the ladder agent searched). Reserved concurrency of 1
to 2 per snake keeps games on the same containers.

## Toolchain (no sudo)

Rust via rustup (`source native/.cargo-env.sh`), plus `cargo-lambda`, which cross-links with zig.
Both come from a throwaway venv:

    uv venv ~/.local/lambda-tools && uv pip install --python ~/.local/lambda-tools/bin/python cargo-lambda
    export PATH=$HOME/.local/lambda-tools/bin:$PATH
    rustup target add aarch64-unknown-linux-gnu x86_64-unknown-linux-gnu

## Build and package

    source native/.cargo-env.sh
    export CARGO_TARGET_DIR=/dev/shm/naz-target-lambda PATH=$HOME/.local/lambda-tools/bin:$PATH
    export CARGO_PROFILE_RELEASE_PANIC=unwind        # lets the handler catch a search panic
    S=/dev/shm/lambda-stage; mkdir -p $S/weights/az-modes
    cp /home/nathan/slinky-serve/az-modes-M-s0/{config.json,params.npz} $S/weights/az-modes/
    # --include is relative to the current directory and keeps that path inside the zip
    cd $S && cargo lambda build --release --arm64 \
        --manifest-path <repo>/native/crates/lambda/Cargo.toml --output-format zip --include weights
    # x86: the same with --x86-64 (the output file is overwritten, copy the arm64 zip first)

Output: `$CARGO_TARGET_DIR/lambda/slinky-lambda/bootstrap.zip`: `bootstrap` at the root and
`weights/az-modes/...` (which is `/var/task/weights/az-modes`). Measured sizes (M network): arm64
binary 2.18 MB, zip 2.28 MB; x86 zip 2.38 MB. Built and checked as ELF aarch64 and x86-64
(glibc 2.17 symbols via zig, fine for `provided.al2023`). The arm64 binary runs on Lambda
(`slinky-native`); the NEON, `sdot` and early-stop code of this version was also run on a
Graviton2 test function on 2026-10-18 (`deploy/lambda-native/README.md`).

Optional speed-up, x86 only: `RUSTFLAGS="-C target-cpu=x86-64-v3"` (AVX2) makes the move about 20 to
25% faster (measured on the PC: standard 4p 14.7 ms against 18.9 ms in the same session) with the
same moves (461/461 Python-golden requests agree). Lambda x86 hosts all have AVX2, but the
function then fails with SIGILL on a CPU without it. The arm64 build is already a
`target-cpu=neoverse-n1` one: cargo-lambda passes `-C target-cpu=neoverse-n1` for `--arm64`.

## Tests and local run

    cargo test -p slinky-lambda --release     # unit, handler, review, runtime, engine_pipe tests
    CARGO_PROFILE_RELEASE_PANIC=unwind cargo test -p slinky-lambda --release   # + the panic test

`tests/handler.rs` (needs the deployed M run dir, `$NAZ_RUN` to change it; the golden part needs
the vectors in `$NAZ_GOLDEN`, both SKIP when missing) checks, through the same `App::handle` the
Lambda calls:

* the four modes' example requests (`/start`, `/move`, `/end`) against `Agent::play` with the
  documented seed and 256 simulations (the engine's default `RESCALE_Q`);
* 461 ladder-shaped requests (4 modes, 2 to 4 snakes, with and without `/start` count) from the
  Python goldens (`review_ladder.jsonl`) with `TIE_SEED` set to the golden seed and
  `RESCALE_Q=none`: all 461 moves equal the Python agent's (also with the x86-64-v3 build);
* `RESCALE_Q` reaching the search: in a 1v1 endgame of a 4-snake standard game where the
  rescaled and the plain search play different moves, the default plays the rescaled move and
  `RESCALE_Q=none` (or a list without `standard`) the plain one;
* routing, 400s, lone snake and 9x9 board answering the safe move, missing weights, `SIMULATIONS=1024`
  against the agent with 1024 simulations, `BUDGET_MS=60` answering within 120 ms, `THREADS=2`.

Local server (builds for the host; a first call compiles, so wait for it):

    WEIGHTS_DIR=/dev/shm/lambda-stage/weights cargo lambda watch -p slinky-lambda --release -P 9001
    curl localhost:9001/slinky
    curl -XPOST localhost:9001/slinky/start -d @start.json
    curl -XPOST localhost:9001/slinky/move  -d @move.json

Checked this way: real `/start`, `/move`, `/end` for standard, duel, royale and constrictor; the
four example moves equal `naz --starting N --seed S` with the seed from the log line; and 40
random ladder requests (17 duel, 11 standard, 7 royale, 5 constrictor) through HTTP equal the
`naz` CLI move for the logged seed (40/40). That was before `RESCALE_Q`: a standard game now needs
`naz --rescale-q` (`slinky-search`'s `tests/naz_cli.rs` checks that its move and root visits are
those of the engine's agent).

## Measurements

All by running the built `bootstrap` against a mock Runtime API that sends real Function-URL
(payload v2) events and times each answer (`tools/mock_runtime.py BOOTSTRAP CPU WARM_N [ENV=V ...]`), M
network, `provided`-style process, one search per request, example requests at turn 0.

x86, the PC (Ryzen 7 5800XT, pinned to one core with `taskset`, other jobs running, SSE2 baseline
build). Warm = event handed out to answer posted, mean of 20 (10 for 1024, 5 for 4096):

| search | standard 4p | duel | royale 4p | constrictor 4p |
|---|---|---|---|---|
| 256 simulations (faithful) | 20.7 ms | 11.2 ms | 20.9 ms | 21.4 ms |
| 1024 simulations | 56.8 | 30.4 | 59.4 | 58.6 |
| 4096 simulations | 226 | 116 | 232 | 234 |
| `BUDGET_MS=350`: simulations reached | 6,263 | 11,949 | 6,323 | 6,134 |
| 1024 simulations, `THREADS=2` (2 cores) | 29.7 | 16.2 | 29.9 | 31.3 |
| `BUDGET_MS=350 THREADS=2`: simulations reached | 11,856 | 16,384 (cap) | 11,841 | 11,443 |

The time budget is exact: every answer took 350.3 ms. Resident memory is small (45 MB at 16,384
simulations).

x86, ThinkPad-class laptop (i7-8550U, 1.8 GHz base, pinned to a core, SSE2 build) as a slower
proxy: 256 simulations 32.1 / 17.4 / 32.8 / 33.4 ms; 1024 simulations 130 / 69 / 132 / 134 ms;
`BUDGET_MS=350` reached 2,727 / 5,323 / 2,696 / 2,630 simulations.

Cold start (process start to the first move answered; includes loading the weights, not Lambda's
own sandbox and runtime-download time, which cannot be measured here): PC 17.5 to 19.7 ms in 10
runs (3.4 ms until the runtime asks for its first event, which covers the 1.6 ms weight load);
laptop 35 ms. On Lambda add the platform's init (typically a few tens to 100+ ms for a small custom
runtime: not measured) once per container.

The same binary benchmarks the CPU it runs on when invoked directly with `{"bench": {}}`
(search ms per move plain / with early stop / with reuse, network us per row on each path):
see `src/bench.rs` and `deploy/lambda-native/README.md`; locally, `cargo run --release -p
slinky-lambda --example lambda_bench -- <run dir>`.

## Lambda cost and timing

What is measured: everything in the tables above, on x86 hosts that are faster (PC) or about as
fast (laptop) as a Lambda vCPU. What is estimated: every arm64 figure except the Graviton2 ones in
Early stop and reuse above (a Lambda test function, 2026-10-18), every Lambda-side figure, and the
prices.

A vCPU share is proportional to memory: 1769 MB is one full vCPU, 3538 MB two; below 1769 MB the
function gets a fractional vCPU and the times above grow in proportion (1024 MB is 0.58 vCPU:
about 1.7x slower). So the useful sizes are 1769 MB (`THREADS=1`) and 3538 MB (`THREADS=2`).
Estimated move time on arm64 at 1 vCPU, assuming a Graviton2 vCPU is 1.0 to 1.6 times the laptop's
time (a Neoverse N1 core at 2.5 GHz against a Skylake mobile core): 256 simulations 30 to 50 ms
(standard 4p; duel half of it), 1024 simulations 130 to 200 ms, `BUDGET_MS=300` about 1,500 to
2,500 simulations. On x86 Lambda the laptop numbers apply to within that same uncertainty, and
the AVX2 build gives back about 20%.

Price (us-east-1 list prices from memory, to be checked: x86 $0.0000166667 and arm64
$0.0000133334 per GB-s, $0.20 per million requests, billed per ms, free tier ignored). Per
million moves, billed time = measured/estimated move time, 4p standard:

| configuration | billed ms | x86 | arm64 |
|---|---|---|---|
| 256 sims, 1769 MB | 21 (PC) to 32 (laptop); arm64 est. 32 to 50 | $0.80 to $1.12 | $0.94 to $1.35 |
| 1024 sims, 1769 MB | 57 (PC) to 130 (laptop); arm64 est. 130 to 200 | $1.84 to $3.94 | $3.19 to $4.81 |
| `BUDGET_MS=300`, 1769 MB | about 305 | $8.98 | $7.23 |
| `BUDGET_MS=300`, 3538 MB, `THREADS=2` | about 305 | $17.76 | $14.25 |

At 100 moves per game that is between $0.0001 and $0.002 per game. arm64 is 20% cheaper per ms,
and on this workload about as fast as the laptop-class x86, so it is the cheaper choice unless the
x86 AVX2 build is used (then x86 is faster per move and costs about the same).

Timing against Battlesnake's 500 ms per move: the time budget counts from the handler's start, so
the HTTP round trip (Battlesnake to the Function URL and back, typically 20 to 100 ms from us-east)
and a cold start come on top. Keep `BUDGET_MS` at 300 or less, or use a fixed `SIMULATIONS` of 1024
(130 to 200 ms estimated on arm64). Two root-parallel trees double the simulations per time (1.9x
measured) but are two trees, not one tree of the double size: their strength relative to a single
1024-simulation tree is not measured.

## Deploy outline for the standalone handler (not used: the engine ships inside `deploy/lambda`)

Mirror `deploy/lambda/deploy.sh` (role, log group with 14-day retention, S3 bucket for the
package, public Function URL, reserved concurrency) with these differences:
`--runtime provided.al2023 --architectures arm64`, `--handler bootstrap`, the zip above, function
name e.g. `slinky-snake-native` (separate from `slinky-snake`), `--memory-size 1769 --timeout 5`
(3538 for `THREADS=2`), and

    Variables={SNAKES=az-256=az-modes:simulations=256,SNAKE_AUTHOR=nathanWolo,SNAKE_HEAD=cosmic-horror,
               SNAKE_TAIL=cosmic-horror,SNAKE_COLOR_AZ_256=#ef4444}

(the same ladder agent as `az-256` on the PC), or for more search
`SNAKES=az-1024=az-modes:simulations=1024`, or `BUDGET_MS=300` with `THREADS` equal to the vCPUs.
Alternatively `cargo lambda deploy slinky-lambda --binary-name slinky-lambda --enable-function-url`
with the same profile. After deploying, send the example requests through the Function URL and
compare the moves with `naz --seed` (plus `--starting N`, and `--rescale-q` for a standard game
unless `RESCALE_Q=none`) for the seeds in the CloudWatch log lines (one JSON line per
event: `by`, `sims`, `mode`, `seed`, `ms`, `error`).
