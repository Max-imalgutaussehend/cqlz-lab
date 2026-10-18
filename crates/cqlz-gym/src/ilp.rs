//! The classic formulation of Cassandra table selection as a mixed-integer program (the NoSE idea): one binary
//! variable per candidate table, one continuous assignment variable per servable (query, table) pair.
//!
//! `min  sum_t open_t * y_t + sum_{q,t} cost_{q,t} * x_{q,t}`
//! `s.t. sum_t x_{q,t} = 1   for every query q,   x_{q,t} <= y_t,   y_t in {0,1}`
//!
//! With `y` fixed, the best `x` is integral (each query takes its cheapest open table), so only `y` is integer.
//! Solved with `microlp`, a pure-Rust simplex with branch and bound.

use crate::problem::Problem;
use crate::solve::Solution;
use good_lp::{
    Expression, ProblemVariables, Solution as _, SolverModel, constraint, microlp, variable,
};
use std::sync::mpsc;
use std::time::Duration;

/// Solves the MILP; `None` if it did not finish within `limit` (the solver thread is left to finish on its own).
pub fn solve(p: &Problem, limit: Duration) -> Option<Solution> {
    // The problem is cheap to snapshot: just the numbers the formulation needs.
    let nq = p.n_queries();
    let nt = p.tables.len();
    let read = p.read.clone();
    let open = p.open.clone();
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut vars = ProblemVariables::new();
        let y: Vec<_> = (0..nt).map(|_| vars.add(variable().binary())).collect();
        let mut x = vec![vec![None; nt]; nq];
        for q in 0..nq {
            for t in 0..nt {
                if read[q][t].is_some() {
                    x[q][t] = Some(vars.add(variable().min(0.0).max(1.0)));
                }
            }
        }
        let mut obj = Expression::from(0.0);
        for t in 0..nt {
            obj += open[t] * y[t];
        }
        for q in 0..nq {
            for t in 0..nt {
                if let (Some(v), Some(c)) = (x[q][t], read[q][t]) {
                    obj += c * v;
                }
            }
        }
        let mut model = vars.minimise(obj).using(microlp);
        for q in 0..nq {
            let mut sum = Expression::from(0.0);
            for t in 0..nt {
                if let Some(v) = x[q][t] {
                    sum += v;
                    model = model.with(constraint!(v <= y[t]));
                }
            }
            model = model.with(constraint!(sum == 1.0));
        }
        let result = model.solve().ok().map(|sol| {
            (0..nt)
                .filter(|&t| sol.value(y[t]) > 0.5)
                .collect::<Vec<usize>>()
        });
        let _ = tx.send(result);
    });
    let set = rx.recv_timeout(limit).ok()??;
    let cost = p.cost(&set)?;
    Some(Solution {
        set,
        cost,
        optimal: true,
    })
}
