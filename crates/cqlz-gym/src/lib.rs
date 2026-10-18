#![allow(clippy::needless_range_loop, clippy::collapsible_if)] // index loops over the cost matrix are intentional
//! Alpha research code: verifier-in-the-loop design of Cassandra tables.
//!
//! `cqlc-core` is the exact verifier: a candidate table may only serve a query if the query compiles
//! against it with **no** diagnostics. Solvers (naive, greedy, exact branch-and-bound, learned policy)
//! pick table sets; the cost model is explicitly synthetic.

pub mod emit;
pub mod ilp;
pub mod policy;
pub mod problem;
pub mod rl;
pub mod solve;
pub mod synth;
pub mod validate;
pub mod workload;
