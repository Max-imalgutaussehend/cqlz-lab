//! Environment, rollouts and REINFORCE training for the table-selection policy.

use crate::policy::{N_FEAT, PolicyConfig, PolicyNet};
use crate::problem::Problem;
use crate::solve::{gain, greedy, unserved_penalty};
use burn::optim::{AdamWConfig, GradientsParams};
use burn::prelude::*;
use burn::tensor::activation::{log_softmax, softmax};
use rand::prelude::*;

/// Sequential table selection on one [`Problem`]. Candidates are the actions; the last action is STOP.
pub struct Env<'a> {
    p: &'a Problem,
    cur: Vec<f64>,
    served: Vec<bool>,
    pub set: Vec<usize>,
    scale: f64,
    pub done: bool,
}

impl<'a> Env<'a> {
    pub fn new(p: &'a Problem, scale: f64) -> Self {
        let nq = p.n_queries();
        Env {
            p,
            cur: (0..nq).map(|q| unserved_penalty(p, q)).collect(),
            served: vec![false; nq],
            set: Vec::new(),
            scale,
            done: false,
        }
    }

    pub fn n_actions(&self) -> usize {
        self.p.tables.len() + 1
    }

    fn all_served(&self) -> bool {
        self.served.iter().all(|&s| s)
    }

    /// Features `[(C+1) * N_FEAT]` and the legality mask `[C+1]`.
    pub fn observe(&self) -> (Vec<f32>, Vec<bool>) {
        let p = self.p;
        let (nt, nq) = (p.tables.len(), p.n_queries() as f64);
        let unserved = self.served.iter().filter(|&&s| !s).count() as f64 / nq;
        let mut f = vec![0f32; (nt + 1) * N_FEAT];
        let mut legal = vec![false; nt + 1];
        for t in 0..nt {
            if self.set.contains(&t) {
                continue;
            }
            let (mut saving, mut newly, mut improved) = (0.0, 0.0, 0.0);
            for q in 0..p.n_queries() {
                if let Some(r) = p.read[q][t] {
                    if r < self.cur[q] {
                        saving += self.cur[q] - r;
                        improved += 1.0;
                        if !self.served[q] {
                            newly += 1.0;
                        }
                    }
                }
            }
            if improved == 0.0 {
                continue;
            }
            legal[t] = true;
            let g = gain(p, &self.cur, t);
            let row = &mut f[t * N_FEAT..(t + 1) * N_FEAT];
            row[0] = (p.open[t] / self.scale) as f32;
            row[1] = (g / self.scale) as f32;
            row[2] = (saving / self.scale) as f32;
            row[3] = (newly / nq) as f32;
            row[4] = (improved / nq) as f32;
            row[5] = unserved as f32;
            row[6] = (self.set.len() as f64 / nq) as f32;
        }
        let stop = &mut f[nt * N_FEAT..];
        stop[5] = unserved as f32;
        stop[6] = (self.set.len() as f64 / nq) as f32;
        stop[7] = 1.0;
        legal[nt] = self.all_served();
        (f, legal)
    }

    pub fn step(&mut self, action: usize) {
        let nt = self.p.tables.len();
        if action == nt {
            self.done = true;
            return;
        }
        for q in 0..self.p.n_queries() {
            if let Some(r) = self.p.read[q][action] {
                if r < self.cur[q] {
                    self.cur[q] = r;
                }
                self.served[q] = true;
            }
        }
        self.set.push(action);
    }

    pub fn cost(&self) -> Option<f64> {
        self.p.cost(&self.set)
    }
}

fn to_tensors(
    feats: &[f32],
    legal: &[bool],
    pad: &[bool],
    b: usize,
    c: usize,
    device: &Device,
) -> (Tensor<3>, Tensor<2>, Tensor<2, Bool>) {
    let f = Tensor::<1>::from_floats(feats, device).reshape([b, c, N_FEAT]);
    let neg: Vec<f32> = legal.iter().map(|&l| if l { 0.0 } else { -1e9 }).collect();
    let m = Tensor::<1>::from_floats(neg.as_slice(), device).reshape([b, c]);
    let pv: Vec<i64> = pad.iter().map(|&x| i64::from(x)).collect();
    let pm = Tensor::<1, Int>::from_ints(pv.as_slice(), device)
        .reshape([b, c])
        .equal_elem(1);
    (f, m, pm)
}

/// One recorded decision for the policy-gradient update.
pub struct Step {
    feats: Vec<f32>,
    legal: Vec<bool>,
    action: usize,
    adv: f32,
}

/// Runs `k` parallel rollouts of the policy on `p`. `sample`: draw from the distribution, else argmax.
/// Returns the final table sets and (if requested) the recorded steps.
pub fn rollouts(
    net: &PolicyNet,
    p: &Problem,
    scale: f64,
    k: usize,
    sample: bool,
    rng: &mut impl Rng,
    device: &Device,
) -> (Vec<Vec<usize>>, Vec<Vec<Step>>) {
    let mut envs: Vec<Env> = (0..k).map(|_| Env::new(p, scale)).collect();
    let mut traces: Vec<Vec<Step>> = (0..k).map(|_| Vec::new()).collect();
    let c = p.tables.len() + 1;
    let max_steps = 2 * p.n_queries() + 2;
    for _ in 0..max_steps {
        let active: Vec<usize> = (0..k).filter(|&i| !envs[i].done).collect();
        if active.is_empty() {
            break;
        }
        let mut feats = Vec::with_capacity(active.len() * c * N_FEAT);
        let mut legal = Vec::with_capacity(active.len() * c);
        for &i in &active {
            let (f, l) = envs[i].observe();
            feats.extend(f);
            legal.extend(l);
        }
        let pad = vec![false; active.len() * c];
        let (ft, mask, pm) = to_tensors(&feats, &legal, &pad, active.len(), c, device);
        let probs = softmax(net.forward(ft, pm) + mask, 1)
            .into_data()
            .try_to_vec::<f32>()
            .expect("f32");
        for (row, &i) in active.iter().enumerate() {
            let pr = &probs[row * c..(row + 1) * c];
            let action = if sample {
                let mut u: f32 = rng.random();
                let mut pick = pr.iter().rposition(|&x| x > 0.0).unwrap_or(c - 1);
                for (a, &x) in pr.iter().enumerate() {
                    if u < x {
                        pick = a;
                        break;
                    }
                    u -= x;
                }
                pick
            } else {
                pr.iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(a, _)| a)
                    .unwrap_or(c - 1)
            };
            traces[i].push(Step {
                feats: feats[row * c * N_FEAT..(row + 1) * c * N_FEAT].to_vec(),
                legal: legal[row * c..(row + 1) * c].to_vec(),
                action,
                adv: 0.0,
            });
            envs[i].step(action);
        }
    }
    let sets = envs.iter().map(|e| e.set.clone()).collect();
    (sets, traces)
}

#[derive(Clone, Debug)]
pub struct TrainOptions {
    pub iterations: usize,
    pub instances_per_iter: usize,
    pub rollouts: usize,
    pub lr: f64,
    pub seed: u64,
    pub entropy: f32,
}

/// Greedy-argmax and best-of-`k` sampled costs of a trained policy on one problem.
pub fn solve_with_policy(
    net: &PolicyNet,
    p: &Problem,
    scale: f64,
    k: usize,
    rng: &mut impl Rng,
    device: &Device,
) -> (f64, f64) {
    let (sets, _) = rollouts(net, p, scale, 1, false, rng, device);
    let argmax = p.cost(&sets[0]).unwrap_or(f64::INFINITY);
    let (sets, _) = rollouts(net, p, scale, k, true, rng, device);
    let best = sets.iter().filter_map(|s| p.cost(s)).fold(argmax, f64::min);
    (argmax, best)
}

/// REINFORCE with a group (per-instance) baseline; reward = -(cost / greedy cost).
pub fn train(
    cfg: &PolicyConfig,
    opts: &TrainOptions,
    mut make: impl FnMut(&mut StdRng) -> Problem,
    mut validate: impl FnMut(&PolicyNet) -> String,
    device: &Device,
) -> PolicyNet {
    let ad = device.clone().autodiff();
    let mut net = cfg.init(&ad);
    let mut optim = AdamWConfig::new().with_weight_decay(0.0).init();
    let mut rng = StdRng::seed_from_u64(opts.seed);
    println!("policy: {} parameters", net.num_params());
    for it in 1..=opts.iterations {
        let infer = net.valid();
        let mut steps: Vec<Step> = Vec::new();
        let mut mean_ratio = 0.0;
        let mut n_ok = 0;
        for _ in 0..opts.instances_per_iter {
            let p = make(&mut rng);
            let g = greedy(&p);
            let (sets, mut traces) =
                rollouts(&infer, &p, g.cost, opts.rollouts, true, &mut rng, device);
            let rewards: Vec<Option<f64>> = sets
                .iter()
                .map(|s| p.cost(s).map(|c| -(c / g.cost)))
                .collect();
            let valid: Vec<f64> = rewards.iter().flatten().copied().collect();
            if valid.len() < 2 {
                continue;
            }
            let mean = valid.iter().sum::<f64>() / valid.len() as f64;
            let sd = (valid.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / valid.len() as f64)
                .sqrt()
                .max(1e-3);
            mean_ratio += -mean;
            n_ok += 1;
            for (tr, r) in traces.iter_mut().zip(&rewards) {
                let Some(r) = r else { continue };
                for s in tr.iter_mut() {
                    s.adv = ((r - mean) / sd) as f32;
                }
            }
            steps.extend(traces.into_iter().flatten());
        }
        if steps.is_empty() {
            continue;
        }
        steps.shuffle(&mut rng);
        for chunk in steps.chunks(192) {
            let b = chunk.len();
            let c = chunk
                .iter()
                .map(|s| s.legal.len())
                .max()
                .expect("non-empty");
            let mut feats = vec![0f32; b * c * N_FEAT];
            let mut legal = vec![false; b * c];
            let mut pad = vec![true; b * c];
            let mut act = vec![0f32; b * c];
            let mut adv = vec![0f32; b];
            for (i, s) in chunk.iter().enumerate() {
                let ci = s.legal.len();
                feats[i * c * N_FEAT..i * c * N_FEAT + ci * N_FEAT].copy_from_slice(&s.feats);
                legal[i * c..i * c + ci].copy_from_slice(&s.legal);
                pad[i * c..i * c + ci].fill(false);
                act[i * c + s.action] = 1.0;
                adv[i] = s.adv;
            }
            let (ft, mask, pm) = to_tensors(&feats, &legal, &pad, b, c, &ad);
            let onehot = Tensor::<1>::from_floats(act.as_slice(), &ad).reshape([b, c]);
            let advt = Tensor::<1>::from_floats(adv.as_slice(), &ad);
            let logp_all = log_softmax(net.forward(ft, pm) + mask, 1);
            let logp = (logp_all.clone() * onehot).sum_dim(1).squeeze_dim::<1>(1);
            let ent = -(logp_all.clone().exp() * logp_all.clamp_min(-30.0))
                .sum_dim(1)
                .squeeze_dim::<1>(1);
            let loss = -(logp * advt).mean() - ent.mean() * opts.entropy;
            let grads = GradientsParams::from_grads(loss.backward(), &net);
            net = optim.step(opts.lr, net, grads);
        }
        if it % 10 == 0 || it == 1 {
            println!(
                "iter {it:>4}/{}  sampled cost / greedy: {:.4}   {}",
                opts.iterations,
                mean_ratio / n_ok.max(1) as f64,
                if it % 50 == 0 {
                    validate(&net.valid())
                } else {
                    String::new()
                }
            );
        }
    }
    net.valid()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::CostModel;
    use crate::workload::Workload;

    #[test]
    fn untrained_policy_rollouts_always_serve_every_query() {
        let w = Workload::from_toml(include_str!("../examples/shop.toml")).unwrap();
        let p = Problem::build(w, CostModel::default(), 64);
        let g = crate::solve::greedy(&p).cost;
        let device = Device::flex();
        let net = PolicyConfig::new().init(&device);
        let mut rng = StdRng::seed_from_u64(1);
        let (sets, _) = rollouts(&net, &p, g, 16, true, &mut rng, &device);
        // Action masking guarantees feasibility without any training.
        assert!(sets.iter().all(|s| p.cost(s).is_some()));
    }

    #[test]
    fn env_stop_is_only_legal_when_everything_is_served() {
        let w = Workload::from_toml(include_str!("../examples/shop.toml")).unwrap();
        let p = Problem::build(w, CostModel::default(), 64);
        let mut env = Env::new(&p, 1.0);
        let n = env.n_actions();
        assert!(!env.observe().1[n - 1]);
        while !env.observe().1[n - 1] {
            let legal = env.observe().1;
            env.step(legal.iter().position(|&l| l).unwrap());
        }
        assert!(env.cost().is_some());
    }

    #[test]
    fn save_load_roundtrip_is_exact() {
        let device = Device::flex();
        let cfg = PolicyConfig::new()
            .with_d_model(16)
            .with_n_heads(2)
            .with_d_ff(32)
            .with_n_layers(1);
        let net = cfg.init(&device);
        let dir = std::env::temp_dir().join(format!("cqlz-gym-test-{}", std::process::id()));
        crate::policy::save(&net, &cfg, &dir).unwrap();
        let loaded = crate::policy::load(&dir, &device).unwrap();
        let f = Tensor::<3>::ones([1, 4, N_FEAT], &device);
        let pad = Tensor::<2, Int>::zeros([1, 4], &device).equal_elem(1);
        let a = net
            .forward(f.clone(), pad.clone())
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let b = loaded
            .forward(f, pad)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(a, b);
        std::fs::remove_dir_all(dir).ok();
    }
}
