//! Partition growth as a birth-death chain.
//!
//! State `n` = live rows in one partition. Inserts arrive at rate `lambda` (rows/s). With a TTL, every
//! live row expires independently after an exponential lifetime with mean `ttl` seconds, so the death
//! rate in state `n` is `n / ttl`. Without TTL the chain is a pure birth process. The size limit `L`
//! is an absorbing state.
//!
//! This models row counts only. It says nothing about tombstone retention (`gc_grace_seconds`),
//! compaction, or bytes; those are out of scope for the alpha.

use rand::Rng;

#[derive(Clone, Copy, Debug)]
pub struct Growth {
    /// Inserted rows per second into this partition.
    pub lambda: f64,
    /// Mean row lifetime in seconds; `None` = rows never expire.
    pub ttl: Option<f64>,
    /// Live-row limit that counts as "too big".
    pub limit: u64,
}

impl Growth {
    /// Exact expected time in seconds to first reach `limit` rows starting from an empty partition.
    /// `f64::INFINITY` when the value overflows (the limit is effectively never reached).
    ///
    /// Birth-death hitting times: `tau_n = 1/lambda + (mu_n/lambda) * tau_{n-1}` is the expected time
    /// to go from `n` to `n+1`, with `mu_n = n/ttl`; the answer is `sum_{n<L} tau_n`.
    pub fn expected_time_to_limit(&self) -> f64 {
        let mu = self.ttl.map_or(0.0, |t| 1.0 / t);
        let mut tau = 1.0 / self.lambda;
        let mut total = 0.0;
        for n in 0..self.limit {
            if n > 0 {
                tau = 1.0 / self.lambda + (n as f64 * mu / self.lambda) * tau;
            }
            total += tau;
            if !total.is_finite() || total > 1e300 {
                return f64::INFINITY;
            }
        }
        total
    }

    /// Steady-state mean live rows (`lambda * ttl`), or `None` without TTL (unbounded growth).
    pub fn steady_state_mean(&self) -> Option<f64> {
        self.ttl.map(|t| self.lambda * t)
    }

    /// Monte-Carlo (Gillespie) estimate of `P(limit reached within horizon seconds)`.
    pub fn prob_within(&self, horizon: f64, runs: usize, rng: &mut impl Rng) -> f64 {
        let mu = self.ttl.map_or(0.0, |t| 1.0 / t);
        let hits = (0..runs)
            .filter(|_| {
                let (mut t, mut n) = (0.0, 0u64);
                loop {
                    let death = n as f64 * mu;
                    let total = self.lambda + death;
                    t += -rng.random::<f64>().ln() / total;
                    if t > horizon {
                        return false;
                    }
                    if rng.random::<f64>() < self.lambda / total {
                        n += 1;
                        if n >= self.limit {
                            return true;
                        }
                    } else {
                        n -= 1;
                    }
                }
            })
            .count();
        hits as f64 / runs as f64
    }

    /// Monte-Carlo mean hitting time (only meaningful when it is small enough to simulate).
    pub fn simulate_mean_time(&self, runs: usize, rng: &mut impl Rng) -> f64 {
        let mu = self.ttl.map_or(0.0, |t| 1.0 / t);
        let mut sum = 0.0;
        for _ in 0..runs {
            let (mut t, mut n) = (0.0, 0u64);
            while n < self.limit {
                let total = self.lambda + n as f64 * mu;
                t += -rng.random::<f64>().ln() / total;
                if rng.random::<f64>() < self.lambda / total {
                    n += 1;
                } else {
                    n -= 1;
                }
            }
            sum += t;
        }
        sum / runs as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn pure_birth_is_limit_over_lambda() {
        let g = Growth {
            lambda: 50.0,
            ttl: None,
            limit: 10_000,
        };
        assert!((g.expected_time_to_limit() - 200.0).abs() < 1e-9);
    }

    #[test]
    fn exact_hitting_time_matches_simulation_with_ttl() {
        let mut rng = StdRng::seed_from_u64(11);
        // Steady-state mean = 4; reaching 9 live rows is possible but not immediate.
        let g = Growth {
            lambda: 2.0,
            ttl: Some(2.0),
            limit: 9,
        };
        let exact = g.expected_time_to_limit();
        let sim = g.simulate_mean_time(40_000, &mut rng);
        assert!(
            (exact - sim).abs() / exact < 0.03,
            "exact {exact} vs simulated {sim}"
        );
    }

    #[test]
    fn far_above_steady_state_is_effectively_never() {
        let g = Growth {
            lambda: 10.0,
            ttl: Some(60.0),
            limit: 100_000,
        };
        assert_eq!(g.steady_state_mean(), Some(600.0));
        assert!(g.expected_time_to_limit().is_infinite() || g.expected_time_to_limit() > 1e100);
    }

    #[test]
    fn horizon_probability_is_monotone_and_consistent() {
        let mut rng = StdRng::seed_from_u64(5);
        let g = Growth {
            lambda: 5.0,
            ttl: None,
            limit: 100,
        }; // takes ~20 s
        let early = g.prob_within(10.0, 5000, &mut rng);
        let late = g.prob_within(40.0, 5000, &mut rng);
        assert!(early < 0.05 && late > 0.95, "{early} {late}");
    }
}
