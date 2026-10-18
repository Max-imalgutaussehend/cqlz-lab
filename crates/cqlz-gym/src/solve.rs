//! Baselines: naive, greedy and exact branch-and-bound.

use crate::problem::Problem;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct Solution {
    pub set: Vec<usize>,
    pub cost: f64,
    /// For the exact solver: proven optimal within the time limit.
    pub optimal: bool,
}

/// One canonical table per query (the first candidates are exactly those), duplicates removed.
pub fn naive(p: &Problem) -> Solution {
    let mut set: Vec<usize> = Vec::new();
    for q in 0..p.n_queries() {
        // candidates 0..n_queries are the canonical tables in query order
        let t = (0..p.tables.len())
            .filter(|&t| p.read[q][t].is_some())
            .min_by(|&a, &b| p.tables[a].columns.len().cmp(&p.tables[b].columns.len()))
            .expect("servable");
        if !set.contains(&t) {
            set.push(t);
        }
    }
    let cost = p.cost(&set).expect("naive serves everything");
    Solution {
        set,
        cost,
        optimal: false,
    }
}

/// Penalty for an unserved query inside greedy/RL: 3× its worst possible read.
pub fn unserved_penalty(p: &Problem, q: usize) -> f64 {
    3.0 * p.read[q].iter().flatten().fold(0.0f64, |m, &c| m.max(c))
}

/// Marginal gain of opening `t` given the current best read per query.
pub fn gain(p: &Problem, cur: &[f64], t: usize) -> f64 {
    let mut g = -p.open[t];
    for (q, c) in cur.iter().enumerate() {
        if let Some(r) = p.read[q][t] {
            if r < *c {
                g += c - r;
            }
        }
    }
    g
}

pub fn greedy(p: &Problem) -> Solution {
    let nq = p.n_queries();
    let mut cur: Vec<f64> = (0..nq).map(|q| unserved_penalty(p, q)).collect();
    let mut served = vec![false; nq];
    let mut set: Vec<usize> = Vec::new();
    loop {
        let best = (0..p.tables.len())
            .filter(|t| !set.contains(t))
            .map(|t| (gain(p, &cur, t), t))
            .max_by(|a, b| a.0.total_cmp(&b.0));
        let all_served = served.iter().all(|&s| s);
        match best {
            Some((g, t)) if g > 0.0 || !all_served => {
                for q in 0..nq {
                    if let Some(r) = p.read[q][t] {
                        if r < cur[q] {
                            cur[q] = r;
                        }
                        served[q] = true;
                    }
                }
                set.push(t);
            }
            _ => break,
        }
    }
    let cost = p.cost(&set).expect("greedy serves everything");
    Solution {
        set,
        cost,
        optimal: false,
    }
}

/// Include/exclude branch-and-bound with an optimistic bound; the greedy solution is the incumbent.
pub fn exact(p: &Problem, limit: Duration) -> Solution {
    let start = Instant::now();
    let inc = greedy(p);
    let nt = p.tables.len();
    let nq = p.n_queries();
    // Branch on tables in order of decreasing potential benefit.
    let mut order: Vec<usize> = (0..nt).collect();
    let potential = |t: usize| (0..nq).filter(|&q| p.read[q][t].is_some()).count();
    order.sort_by_key(|&t| std::cmp::Reverse(potential(t)));
    let min_read: Vec<f64> = (0..nq)
        .map(|q| {
            p.read[q]
                .iter()
                .flatten()
                .fold(f64::INFINITY, |m, &c| m.min(c))
        })
        .collect();
    // suffix_min[i][q]: best read of q among order[i..]
    let mut suffix_min = vec![vec![f64::INFINITY; nq]; nt + 1];
    for i in (0..nt).rev() {
        for q in 0..nq {
            suffix_min[i][q] =
                suffix_min[i + 1][q].min(p.read[q][order[i]].unwrap_or(f64::INFINITY));
        }
    }
    let _ = min_read;
    struct Ctx<'a> {
        p: &'a Problem,
        order: &'a [usize],
        suffix_min: &'a [Vec<f64>],
        best: Solution,
        start: Instant,
        limit: Duration,
        timed_out: bool,
        nodes: u64,
    }
    fn rec(c: &mut Ctx, i: usize, chosen: &mut Vec<usize>, cur: &mut Vec<f64>, open_sum: f64) {
        c.nodes += 1;
        if c.nodes.is_multiple_of(1024) && c.start.elapsed() > c.limit {
            c.timed_out = true;
        }
        if c.timed_out {
            return;
        }
        let nq = cur.len();
        // optimistic bound
        let mut lb = open_sum;
        for q in 0..nq {
            let r = cur[q].min(c.suffix_min[i][q]);
            if !r.is_finite() {
                return; // some query can never be served on this branch
            }
            lb += r;
        }
        if lb >= c.best.cost - 1e-9 {
            return;
        }
        if i == c.order.len() {
            if cur.iter().all(|x| x.is_finite()) {
                c.best = Solution {
                    set: chosen.clone(),
                    cost: lb,
                    optimal: false,
                };
            }
            return;
        }
        let t = c.order[i];
        // include t
        let saved = cur.clone();
        for q in 0..nq {
            if let Some(r) = c.p.read[q][t] {
                if r < cur[q] {
                    cur[q] = r;
                }
            }
        }
        chosen.push(t);
        rec(c, i + 1, chosen, cur, open_sum + c.p.open[t]);
        chosen.pop();
        *cur = saved;
        // exclude t
        rec(c, i + 1, chosen, cur, open_sum);
    }
    let mut ctx = Ctx {
        p,
        order: &order,
        suffix_min: &suffix_min,
        best: inc,
        start,
        limit,
        timed_out: false,
        nodes: 0,
    };
    let mut cur = vec![f64::INFINITY; nq];
    rec(&mut ctx, 0, &mut Vec::new(), &mut cur, 0.0);
    let mut best = ctx.best;
    best.optimal = !ctx.timed_out;
    best
}
