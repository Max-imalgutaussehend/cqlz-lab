//! Alpha research code: Markov models for CQL workloads.
//!
//! * [`session`]: first-order Markov chain over normalised query *shapes* (lexed with `cqlc-core`),
//!   fitted from session logs, used to generate synthetic workloads and scored by held-out perplexity.
//! * [`growth`]: birth-death chain for partition size under inserts and TTL expiry, with exact expected
//!   time to a size limit and a Gillespie simulator to cross-check it.

pub mod growth;
pub mod session;
