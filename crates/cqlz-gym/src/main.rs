use anyhow::{Context, Result};
use burn::prelude::*;
use clap::{Parser, Subcommand};
use cqlz_gym::policy::{self, PolicyConfig};
use cqlz_gym::problem::{CostModel, Problem};
use cqlz_gym::rl::{self, Env, TrainOptions};
use cqlz_gym::solve::{exact, greedy, naive};
use cqlz_gym::synth::{self, Size};
use cqlz_gym::workload::Workload;
use rand::{SeedableRng, rngs::StdRng};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const HF_REPO: &str = "Max-imumlikelihood/cqlz-gym-policy";

/// Alpha research tool: verified, learned Cassandra table design.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Design tables for a workload (TOML) and print verified CQL.
    Solve {
        workload: PathBuf,
        /// Policy directory, or `hf` to fetch it from Hugging Face. Omit to skip the learned solver.
        #[arg(long)]
        policy: Option<String>,
        /// Time limit for the exact branch-and-bound baseline.
        #[arg(long, default_value_t = 5)]
        exact_seconds: u64,
        /// Sampled rollouts for the learned best-of-k.
        #[arg(long, default_value_t = 32)]
        k: usize,
    },
    /// Train the policy with REINFORCE on random workloads.
    Train {
        #[arg(long, default_value = "artifacts")]
        out: PathBuf,
        #[arg(long, default_value_t = 300)]
        iterations: usize,
        #[arg(long, default_value_t = 8)]
        instances: usize,
        #[arg(long, default_value_t = 8)]
        rollouts: usize,
        #[arg(long, default_value_t = 1e-3)]
        lr: f64,
        #[arg(long, default_value_t = 1)]
        seed: u64,
    },
    /// Compare naive / greedy / exact / learned on fresh random workloads of three sizes.
    Eval {
        #[arg(long)]
        policy: String,
        #[arg(long, default_value_t = 30)]
        instances: usize,
        #[arg(long, default_value_t = 3)]
        exact_seconds: u64,
        #[arg(long, default_value_t = 100)]
        seed: u64,
    },
    /// Environment throughput: verification cost and rollout steps per second per core.
    Bench {
        #[arg(long, default_value_t = 20)]
        instances: usize,
    },
}

fn policy_dir(spec: &str) -> Result<PathBuf> {
    if spec != "hf" {
        return Ok(spec.into());
    }
    let root = std::env::var_os("CQLZ_GYM_CACHE")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(&std::env::var_os("HOME").unwrap_or_default()).join(".cache/cqlz-gym")
        });
    let dir = root.join(HF_REPO.replace('/', "--"));
    if !(dir.join(policy::WEIGHTS_FILE).is_file() && dir.join(policy::CONFIG_FILE).is_file()) {
        std::fs::create_dir_all(&dir)?;
        let base = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());
        for f in [policy::CONFIG_FILE, policy::WEIGHTS_FILE] {
            let url = format!("{}/{HF_REPO}/resolve/main/{f}", base.trim_end_matches('/'));
            let resp = ureq::get(&url)
                .call()
                .with_context(|| format!("downloading {url}"))?;
            let tmp = dir.join(format!("{f}.part"));
            std::io::copy(
                &mut resp.into_body().into_reader(),
                &mut std::fs::File::create(&tmp)?,
            )?;
            std::fs::rename(tmp, dir.join(f))?;
        }
    }
    Ok(dir)
}

fn fmt_cost(c: f64) -> String {
    format!("{c:>12.2}")
}

fn solve(path: &Path, policy: Option<&str>, exact_s: u64, k: usize) -> Result<()> {
    let w = Workload::from_toml(
        &std::fs::read_to_string(path).with_context(|| path.display().to_string())?,
    )?;
    let t0 = Instant::now();
    let p = Problem::build(w, CostModel::default(), 256);
    println!(
        "{} candidate tables, {} verifier calls to cqlc-core, {:.0} ms",
        p.tables.len(),
        p.verifier_calls,
        t0.elapsed().as_secs_f64() * 1e3
    );
    if !p.unservable().is_empty() {
        anyhow::bail!("no valid table can serve: {}", p.unservable().join(", "));
    }
    let nv = naive(&p);
    let gr = greedy(&p);
    let ex = exact(&p, Duration::from_secs(exact_s));
    println!(
        "naive (one table per query)   cost {}  tables {}",
        fmt_cost(nv.cost),
        nv.set.len()
    );
    println!(
        "greedy                        cost {}  tables {}",
        fmt_cost(gr.cost),
        gr.set.len()
    );
    println!(
        "exact B&B ({})        cost {}  tables {}",
        if ex.optimal { "optimal " } else { "time-out" },
        fmt_cost(ex.cost),
        ex.set.len()
    );
    let mut best = ex.clone();
    if let Some(spec) = policy {
        let device = Device::flex();
        let net = policy::load(&policy_dir(spec)?, &device)?;
        let mut rng = StdRng::seed_from_u64(1);
        let (a, bk) = rl::solve_with_policy(&net, &p, gr.cost, k, &mut rng, &device);
        println!("learned policy (argmax)       cost {}", fmt_cost(a));
        println!("learned policy (best of {k})   cost {}", fmt_cost(bk));
    }
    if gr.cost < best.cost {
        best = gr;
    }
    let design = p.render(&best.set);
    println!("\n-- design (cost {:.2}) --\n{design}", best.cost);
    // Re-verify the printed design end to end.
    let (schema, queries): (Vec<_>, Vec<_>) = design.lines().partition(|l| l.starts_with("CREATE"));
    let c = cqlc_core::compile(
        &[cqlc_core::Input::new("schema.cql", schema.join("\n"))],
        &[cqlc_core::Input::new("queries.cql", queries.join("\n"))],
    );
    println!(
        "-- cqlc-core re-check of the design: {} diagnostic(s)",
        c.diagnostics.len()
    );
    Ok(())
}

fn eval(policy: &str, n: usize, exact_s: u64, seed: u64) -> Result<()> {
    let device = Device::flex();
    let net = policy::load(&policy_dir(policy)?, &device)?;
    let sizes = [
        (
            "small  (4-7 queries)",
            Size {
                entities: (1, 2),
                queries: (4, 7),
            },
        ),
        (
            "medium (8-16 queries)",
            Size {
                entities: (2, 4),
                queries: (8, 16),
            },
        ),
        (
            "large  (25-50 queries)",
            Size {
                entities: (3, 6),
                queries: (25, 50),
            },
        ),
        (
            "xlarge (80-150 queries)",
            Size {
                entities: (6, 8),
                queries: (80, 150),
            },
        ),
    ];
    println!(
        "{:<24} {:>8} {:>8} {:>8} {:>9} {:>9} {:>8} {:>8} {:>8} {:>8}",
        "size",
        "naive",
        "greedy",
        "exact",
        "learn-am",
        "learn-b32",
        "exact-ok",
        "t_ex(s)",
        "t_am(s)",
        "t_b32(s)"
    );
    for (label, size) in sizes {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut sums = [0.0f64; 5];
        let (mut ok, mut t_ex, mut t_am, mut t_nn, mut count) = (0, 0.0, 0.0, 0.0, 0);
        while count < n {
            let w = synth::workload(&mut rng, size);
            let p = Problem::build(w, CostModel::default(), 256);
            if !p.unservable().is_empty() {
                continue;
            }
            let nv = naive(&p);
            let gr = greedy(&p);
            let t = Instant::now();
            let ex = exact(&p, Duration::from_secs(exact_s));
            t_ex += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let _ = rl::argmax_cost(&net, &p, gr.cost, &mut rng, &device);
            t_am += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let (am, bk) = rl::solve_with_policy(&net, &p, gr.cost, 32, &mut rng, &device);
            t_nn += t.elapsed().as_secs_f64();
            let best = [nv.cost, gr.cost, ex.cost, am, bk]
                .into_iter()
                .fold(f64::INFINITY, f64::min);
            for (s, c) in sums.iter_mut().zip([nv.cost, gr.cost, ex.cost, am, bk]) {
                *s += c / best;
            }
            ok += usize::from(ex.optimal);
            count += 1;
        }
        let m = |i: usize| sums[i] / n as f64;
        println!(
            "{label:<24} {:>8.3} {:>8.3} {:>8.3} {:>9.3} {:>9.3} {:>5}/{:<2} {:>8.2} {:>8.2} {:>8.2}",
            m(0),
            m(1),
            m(2),
            m(3),
            m(4),
            ok,
            n,
            t_ex / n as f64,
            t_am / n as f64,
            t_nn / n as f64
        );
    }
    println!(
        "\n(values: mean cost / best cost found by any method on that instance; 1.000 = always best)"
    );
    Ok(())
}

fn bench(n: usize) {
    let mut rng = StdRng::seed_from_u64(7);
    let (mut verif, mut calls, mut steps, mut t_env) = (0.0, 0usize, 0usize, 0.0);
    for _ in 0..n {
        let w = synth::workload(
            &mut rng,
            Size {
                entities: (3, 6),
                queries: (25, 50),
            },
        );
        let t = Instant::now();
        let p = Problem::build(w, CostModel::default(), 256);
        verif += t.elapsed().as_secs_f64();
        calls += p.verifier_calls;
        if !p.unservable().is_empty() {
            continue;
        }
        let g = greedy(&p).cost;
        let t = Instant::now();
        for _ in 0..20 {
            let mut env = Env::new(&p, g);
            use rand::seq::IndexedRandom;
            while !env.done {
                let (_, legal) = env.observe();
                let opts: Vec<usize> = (0..legal.len()).filter(|&i| legal[i]).collect();
                env.step(*opts.choose(&mut rng).expect("an action is always legal"));
                steps += 1;
            }
        }
        t_env += t.elapsed().as_secs_f64();
    }
    println!(
        "verification: {:.1} us per cqlc-core call ({} calls, {:.2} s total)",
        verif / calls as f64 * 1e6,
        calls,
        verif
    );
    println!(
        "environment (observe + step, random policy, no network): {:.0} steps/s on one core ({} steps)",
        steps as f64 / t_env,
        steps
    );
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Solve {
            workload,
            policy,
            exact_seconds,
            k,
        } => solve(&workload, policy.as_deref(), exact_seconds, k)?,
        Cmd::Train {
            out,
            iterations,
            instances,
            rollouts,
            lr,
            seed,
        } => {
            let device = if cfg!(feature = "wgpu") {
                Device::default()
            } else {
                Device::flex()
            };
            let cfg = PolicyConfig::new();
            let opts = TrainOptions {
                iterations,
                instances_per_iter: instances,
                rollouts,
                lr,
                seed,
                entropy: 0.01,
            };
            let size = Size {
                entities: (1, 4),
                queries: (4, 30),
            };
            let make = move |rng: &mut StdRng| loop {
                let p = Problem::build(synth::workload(rng, size), CostModel::default(), 160);
                if p.unservable().is_empty() {
                    return p;
                }
            };
            let dev = device.clone();
            let validate = move |net: &policy::PolicyNet| {
                let mut rng = StdRng::seed_from_u64(999);
                let (mut r, mut n) = (0.0, 0);
                while n < 10 {
                    let p =
                        Problem::build(synth::workload(&mut rng, size), CostModel::default(), 160);
                    if !p.unservable().is_empty() {
                        continue;
                    }
                    let g = greedy(&p).cost;
                    let (am, _) = rl::solve_with_policy(net, &p, g, 1, &mut rng, &dev);
                    r += am / g;
                    n += 1;
                }
                format!("[val argmax / greedy = {:.4}]", r / n as f64)
            };
            let net = rl::train(&cfg, &opts, make, validate, &device);
            policy::save(&net, &cfg, &out)?;
            println!("saved {}", out.join(policy::WEIGHTS_FILE).display());
        }
        Cmd::Eval {
            policy,
            instances,
            exact_seconds,
            seed,
        } => eval(&policy, instances, exact_seconds, seed)?,
        Cmd::Bench { instances } => bench(instances),
    }
    Ok(())
}
