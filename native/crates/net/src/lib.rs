//! slinky-net: weights loading and the forward pass of slinky's multi-mode AlphaZero network.
//!
//! Ports `az_modes.network` (`baselines/az_modes.py:198`), `unified.torso`/`conv_features`
//! (`src/slinky/unified.py:431-470`) and the `params.npz` / `config.json` loaders
//! (`unified.py:666-712`); see `native-az/SPEC.md` sections 1, 2 and 8. No JAX, no BLAS, no
//! unsafe, no panics on bad files or inputs (every failure is a [`NetError`]).
//!
//! # Public API
//!
//! ```no_run
//! use slinky_net::{Network, OBS_LEN, NUM_FEATURES};
//! let net = Network::load("/path/to/run-dir")?;          // config.json + params.npz
//! let mut scratch = net.scratch();                      // reusable buffers, one per thread
//! let (obs, feats) = (vec![0f32; 2 * OBS_LEN], vec![0f32; 2 * NUM_FEATURES]);
//! let mut logits = [[0f32; 4]; 2];
//! let mut values = [0f32; 2];
//! net.forward_into(&mut scratch, &obs, &feats, &mut logits, &mut values)?;   // no allocation
//! let out = net.forward(&obs, &feats)?;                 // allocating convenience: Output { logits, values }
//! # Ok::<(), slinky_net::NetError>(())
//! ```
//!
//! * [`Network::load`]`(run_dir) -> Result<Network, NetError>`; also
//!   [`Network::from_npz_bytes`]`(NetConfig, &[u8])` and [`Network::from_arrays`]`(NetConfig, HashMap<String, Array>)`.
//! * [`Network::forward_into`]`(&self, &mut Scratch, obs: &[f32], feats: &[f32], logits: &mut [[f32; 4]], values: &mut [f32]) -> Result<(), NetError>`;
//!   the batch size is `values.len()`; `obs` is `[B][21][21][13]` (NHWC, [`OBS_LEN`] floats per row),
//!   `feats` is `[B][7]` as produced by `modes.mode_features`. Rows are independent.
//! * [`Network::forward`]`(&self, obs, feats) -> Result<Output, NetError>`.
//! * [`Network::forward_boards_into`]`(&self, &mut Scratch, boards: &[BoardInput], feats, logits,
//!   values)`: the same pass on structured boards (the window's offset, own health and length,
//!   and the content entries: `slinky_obs::SparseObs`), which skips building the dense window
//!   and adds the three whole-board planes from precomputed per-class sums. Outputs equal
//!   `forward_into`'s up to float summation order: typically ~2e-7, at most a few 1e-6 (the
//!   same order as the Rust-vs-XLA difference), so it can flip a rare near-tie (about 1 search
//!   in 15,000 to 55,000); the search uses this one.
//! * [`Network::config`]`() -> &NetConfig`: includes `c_puct`, `fpu_reduction`, `draw_value`,
//!   `hide_constrictor_food`, `game_modes` for the search and observation crates.
//! * [`Network::num_params`], [`Network::scratch`].
//! * [`Network::quantize_int8`]`(&mut self, on: bool) -> Result<(), NetError>`: off by default,
//!   changes the outputs (every layer after the first convolution in int8, with dot-product
//!   kernels where the CPU has them: [`Int8Kernel`], [`Network::int8_kernel`],
//!   [`Network::set_int8_kernel`]).
//! * [`Network::store_hidden_bf16`]`(&mut self, on: bool)`: experimental, off by default (the
//!   hidden layer's weights rounded to bfloat16, half the memory of the largest matrix; outputs
//!   off by about 2e-3).
//!
//! # Speed
//!
//! The forward pass works on compressed non-zero activations and skips pixels that only see
//! zero padding (see the `network` module docs). `cargo run --release -p slinky-net --example
//! bench -- <run dir> <golden cases.npz>` measures it: for the M net on real states, one x86-64
//! core (Ryzen 7 5800XT, SSE2 baseline build) takes about 18 us per board at batch 1, 2, 4 or 8
//! (the first plain version took 50 us).
//!
//! The inner loops run on 4-lane vectors (the private `simd` module: NEON on aarch64, SSE2 on
//! x86-64, plain arrays elsewhere), with one multiply-add per lane exactly as the scalar code
//! rounds it, so the outputs do not depend on the target's vector width. On the dev box's
//! Xeon, the network on the search's own nodes takes about 30 us per row on the structured
//! path against 35 us dense; on Graviton2 (AWS Lambda, 2026-10-18) about 39-42 us against
//! 46-51 us, 15-18% faster (`cargo run --release -p slinky-lambda --example lambda_bench --
//! <run dir>`, which also runs on the Lambda itself: `deploy/lambda-native/README.md`).
//!
//! [`Network::quantize_int8`] runs the layers after the first convolution on int8 dot-product
//! instructions. It is off in every serving path, it costs accuracy, and its strength is
//! untested (below). On the deployed M net the outputs move by a mean |dlogit| of 0.02 to 0.06 (at
//! most about 0.7) and a mean |dvalue| of 0.007 to 0.022 with a consistent positive bias, and
//! the greedy (argmax) move flips on 2.6% to 4.9% of the nodes. On Graviton2 the argmax flips on
//! 2% to 4% of the rows (mean |dlogit| about 0.02, constrictor 0.056 to 0.069, at most 0.57),
//! and a 256-simulation search plays the same seat-0 move as `f32` in 184 of 192 and 46 of 48
//! positions. Strength is untested: 192 duel games at equal 256 simulations gave 0 +/- 49 Elo,
//! which cannot tell int8 from `f32`, and nothing else has been played.
//!
//! Speed: on Graviton2 the runtime selects the `sdot` kernel, which gave the same error
//! statistics as the plain Rust int8 kernel in all four modes, and a row takes about 33-35 us
//! against 39-42 us for structured `f32`, 1.17 times faster per row (the roughly 50% more
//! simulations a 200 ms search reached with int8 there is not a like-for-like figure: the
//! int8 search played other positions). The plain Rust kernel is slower than `f32` where it is
//! the fallback: about 3 times on Graviton2 (126-130 us per row) and about 10 times on x86-64
//! without AVX-VNNI (Ryzen 7 5800XT, i7-8550U). The AVX-VNNI kernel has never run on any of
//! our test machines: the figures it was written against (16-17 us per row instead of 25-27 on
//! a Xeon, 1.5 to 1.7 times the simulations in 200 ms) are from the author's box and were not
//! reproduced.
//!
//! The raw feature vector goes in: for `features = "no_mode"` the network masks the one-hot and
//! shrink phase itself, for `"none"` it ignores the features. Food hiding in constrictor
//! (`hide_constrictor_food`) is the observation builder's job. Only `compute_dtype = "float32"`
//! is implemented; other dtypes are refused at load time.

mod config;
mod error;
mod network;
pub mod npy;
mod quant;
mod simd;

pub use config::{
    FeatureMode, MODE_NAMES, NUM_ACTIONS, NUM_FEATURES, NetConfig, OBS_CHANNELS, OBS_COLS, OBS_LEN,
    OBS_ROWS,
};
pub use error::NetError;
pub use network::{
    BoardInput, CH_ON_BOARD, CH_OWN_HEALTH, CH_OWN_LENGTH, Network, Output, Scratch,
};
pub use npy::Array;
pub use quant::Int8Kernel;
