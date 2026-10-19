# The native snakes as their own Lambda function

`slinky-native` (arm64, `provided.al2023`, 1769 MB = one vCPU) runs the Rust handler of
`native/crates/lambda` (binary `slinky-lambda` as `bootstrap`, settings in that crate's README)
with the az_modes weights. One Function URL, one path per snake:

| path | search |
|---|---|
| `/az-256` | 256 simulations (the faithful port of the ladder agent); arena name AlphaSlinky-256 |
| `/az-t200` | as many simulations as fit in 200 ms from the Function URL's receipt of the request (cap 16,384); arena name AlphaSlinky |

Why a function of its own: the Python function `slinky-snake` (`deploy/lambda`) takes ~410 ms
to start a container (Python and numpy), and every container it starts in the middle of a game
(concurrent games scale it out) answered past the arena's 500 ms: 3 timeouts in its first
1,346 ladder moves. A Rust runtime starts in tens of ms. Lambda bills per request and duration,
so a second function costs nothing more.

    deploy/lambda-native/build.sh                         # dist/native.zip (deploy.sh runs it)
    AWS_PROFILE=slinky deploy/lambda-native/deploy.sh     # create or update, URL, smoke test

Change the line-up or the think time with `SNAKES` (e.g.
`SNAKES=az-256=az-modes:simulations=256,az-t250=az-modes:budget_ms=250`, plus that snake's
`SNAKE_COLOR_`/`SNAKE_HEAD_`/`SNAKE_TAIL_` variables). Each move logs its `ms`, `sims`,
`deadline_ms`, `cut` and `reused` to CloudWatch (`/aws/lambda/slinky-native`):

* `reused` is the number of nodes whose network evaluation came from the game's previous tree
  (`REUSE`, on by default).
* With the early stop (`EARLY_STOP`, on by default) `sims` below the simulation count is normal
  for `/az-256` (256), and `sims` is 0 on a forced move (one legal move). `/az-t200` searches a
  forced move to its deadline while `REUSE` is on, so its `sims` is 0 on a forced move only with
  `REUSE=0`.
* `cut` means only that the deadline or the memory cap stopped the search. It no longer counts
  a search that ended early because its move was decided or its tree solved (`Solved` stops).
* With `REUSE` a snake holds up to 8 search trees (the current and the previous tree of each of
  its 4 most recent games): an estimated worst case of about 360 MB for `/az-t200`, which fits
  the 1,769 MB.

## Switching the speed-ups off, and rolling back

The early stop and the tree reuse never change a move; they are on by default and the function
can run without them, without a rebuild. `deploy.sh` passes `EARLY_STOP` and `REUSE` from the
shell to the function's environment like `THREADS`. The engine accepts only `0` (off) and `1`
(on); anything else would fail the function at init, so `deploy.sh` refuses it.

    EARLY_STOP=0 REUSE=0 SKIP_BUILD=1 AWS_PROFILE=slinky deploy/lambda-native/deploy.sh   # both off
    EARLY_STOP=1 REUSE=1 SKIP_BUILD=1 AWS_PROFILE=slinky deploy/lambda-native/deploy.sh   # both on again

Either one can be set alone. A deployed value persists across later deploys (the script keeps the
deployed environment and only overwrites the variables set in the shell), so turning them back on
needs an explicit `=1`; a plain `deploy.sh` leaves a `0` in place. `SKIP_BUILD=1` deploys the
`dist/native.zip` that is already there. The current values:
`aws --region us-east-2 lambda get-function-configuration --function-name slinky-native --query Environment.Variables`.
A configuration change starts fresh containers, so do it while no games are being played.

To roll back to a previous build, keep its zip (`build.sh` overwrites `dist/native.zip` and prints
the zip's sha256; copy the live one aside before the next build), put it back as
`dist/native.zip` and deploy with `SKIP_BUILD=1`:

    cp deploy/lambda-native/dist/native-<previous>.zip deploy/lambda-native/dist/native.zip
    SKIP_BUILD=1 AWS_PROFILE=slinky deploy/lambda-native/deploy.sh

A build from before the speed-ups ignores `EARLY_STOP` and `REUSE` (an unknown variable is not an
error).

## Benchmark on the function's own CPU

The binary answers a direct invocation `{"bench": {...}}` (never a Function URL request) with
a speed report from the Graviton it runs on: per mode, ms per 256-simulation move plain, with
the early stop and with the tree reuse, the network's microseconds per row on the dense,
structured, bfloat16 and int8 paths (with the int8 outputs' error and move agreement), and the
simulations a 200 ms search reaches with and without the tree reuse, and with the reuse on the
int8 network (`native/crates/lambda/src/bench.rs`). It takes about a minute, so
raise the timeout for the run, while no games are being played (a configuration change starts
fresh containers), and put it back:

    aws --region us-east-2 lambda update-function-configuration --function-name slinky-native --timeout 120
    aws --region us-east-2 lambda wait function-updated --function-name slinky-native
    aws --region us-east-2 lambda invoke --function-name slinky-native --cli-read-timeout 150 \
        --cli-binary-format raw-in-base64-out --payload '{"bench": {}}' bench.json
    aws --region us-east-2 lambda update-function-configuration --function-name slinky-native --timeout 3

The int8 and bfloat16 networks exist only in this benchmark: no served snake uses them. Measured
on this hardware (a test function, Graviton2, 1,769 MB, 2026-10-18; the runtime selected the
`sdot` kernel), network per row: dense `f32` about 46-51 us, structured `f32` about 39-42 us,
structured int8 with `sdot` about 33-35 us (1.17 times faster than structured `f32`), the plain
Rust int8 kernel about 126-130 us (3 times slower than `f32`). `sdot` and the plain kernel gave
identical error statistics in all four modes. Against `f32`, the int8 network's argmax flips on
2-4% of rows, the mean |logit error| is about 0.02 (constrictor 0.056-0.069) and at most 0.57,
and at 256 simulations the seat-0 move equals `f32`'s in 184/192 and 46/48 positions. On the
deployed M net the x86 review measured mean |dlogit| 0.02-0.06 (up to about 0.7), mean |dvalue|
0.007-0.022 with a consistent positive bias, and greedy flips on 2.6-4.9% of nodes. Strength is
untested beyond 0 +/- 49 Elo at equal 256 simulations (192 duel games). The AVX-VNNI kernel has
never run on a test machine, and x86 without it falls back to the plain kernel, about 10 times
slower than `f32`. The roughly 50% more simulations a 200 ms search got with int8 is not a
like-for-like figure: int8 played other positions, and the gain per row is 1.17 times.

Options: `{"bench": {"snake": "az-256", "positions": 12, "seed": 1, "modes": ["duel"],
"budget_ms": 200, "turns": 10, "int8_kernel": "scalar"}}` (`int8_kernel` forces the plain
Rust int8 kernel, to compare with `sdot`). The
report stops early (`"cut_short": true`) 300 ms before the invocation's deadline. The same
report on another machine: `cargo run --release -p slinky-lambda --example lambda_bench -- <run dir>`.
