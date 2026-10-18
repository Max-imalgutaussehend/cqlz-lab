use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use cqlz_markov::growth::Growth;
use cqlz_markov::session::{SessionModel, parse_log};
use rand::{SeedableRng, rngs::StdRng};
use std::path::PathBuf;

/// Alpha research tool: Markov models for CQL workloads.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Fit a session model from a log (`session_id<TAB>statement` per line) and write model.json.
    Fit {
        log: PathBuf,
        #[arg(short, long, default_value = "model.json")]
        out: PathBuf,
        #[arg(long, default_value_t = 0.1)]
        alpha: f64,
        /// Fraction of sessions held out to report perplexity vs an i.i.d. baseline.
        #[arg(long, default_value_t = 0.2)]
        holdout: f64,
    },
    /// Generate synthetic sessions from a fitted model (one `session<TAB>shape` line per statement).
    Generate {
        model: PathBuf,
        #[arg(short, long, default_value_t = 100)]
        sessions: usize,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Partition growth: expected time until a partition reaches `--limit` live rows.
    Growth {
        /// Inserted rows per second into the partition.
        #[arg(long)]
        lambda: f64,
        /// Mean row lifetime in seconds (omit for no TTL).
        #[arg(long)]
        ttl: Option<f64>,
        #[arg(long)]
        limit: u64,
        /// Also estimate P(limit reached within this many seconds) by simulation.
        #[arg(long)]
        horizon: Option<f64>,
    },
}

fn human(secs: f64) -> String {
    if !secs.is_finite() || secs > 1e12 {
        return "effectively never".into();
    }
    let d = secs / 86400.0;
    if d >= 365.0 {
        format!("{:.1} years", d / 365.0)
    } else if d >= 1.0 {
        format!("{d:.1} days")
    } else {
        format!("{:.1} s", secs)
    }
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Fit {
            log,
            out,
            alpha,
            holdout,
        } => {
            let sessions = parse_log(
                &std::fs::read_to_string(&log).with_context(|| log.display().to_string())?,
            );
            let cut = ((sessions.len() as f64) * (1.0 - holdout)).round() as usize;
            let (train, test) = sessions.split_at(cut.clamp(1, sessions.len()));
            let model = SessionModel::fit(train, alpha)?;
            if !test.is_empty() {
                println!(
                    "held-out bits/statement: markov {:.3}  vs  i.i.d. {:.3}   ({} train / {} test sessions, {} shapes)",
                    model.bits_per_statement(test),
                    model.unigram_bits(train, test),
                    train.len(),
                    test.len(),
                    model.states.len()
                );
            }
            std::fs::write(
                &out,
                serde_json::to_string_pretty(&SessionModel::fit(&sessions, alpha)?)?,
            )?;
            println!("wrote {}", out.display());
        }
        Cmd::Generate {
            model,
            sessions,
            seed,
        } => {
            let m: SessionModel = serde_json::from_str(&std::fs::read_to_string(model)?)?;
            let mut rng = StdRng::seed_from_u64(seed);
            for i in 0..sessions {
                for s in m.sample(&mut rng, 1000) {
                    println!("g{i}\t{s}");
                }
            }
        }
        Cmd::Growth {
            lambda,
            ttl,
            limit,
            horizon,
        } => {
            let g = Growth { lambda, ttl, limit };
            match g.steady_state_mean() {
                Some(m) => println!("steady-state live rows: {m:.0}"),
                None => println!("no TTL: the partition grows without bound"),
            }
            println!(
                "expected time to {limit} live rows: {}",
                human(g.expected_time_to_limit())
            );
            if let Some(h) = horizon {
                let p = g.prob_within(h, 20_000, &mut StdRng::seed_from_u64(1));
                println!(
                    "P(reached within {}): {:.4}  (Monte-Carlo, 20k runs)",
                    human(h),
                    p
                );
            }
        }
    }
    Ok(())
}
