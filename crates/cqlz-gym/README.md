# cqlz-gym (alpha)

**Verifier-in-the-loop design of Cassandra tables.** Part of the **cqlz family** research crates.

Given entities and a query workload (TOML), `cqlz-gym` proposes a set of denormalised tables so that every
query is a clean single-partition read, and prints the `CREATE TABLE` statements plus the queries rewritten
against them. **`cqlc-core` is the exact verifier:** a candidate table may only serve a query if the query
compiles against it with *no* diagnostic, so every proposed design is valid by construction. Solvers: a naive
baseline, greedy, an exact branch-and-bound, and a small Transformer policy (Burn) trained with REINFORCE.

```sh
cargo install cqlz-gym
cqlz-gym solve examples/shop.toml                       # naive / greedy / exact + verified design
cqlz-gym solve examples/shop.toml --policy hf           # also the learned policy (weights from Hugging Face)
cqlz-gym train --iterations 1000 --out artifacts        # train your own
cqlz-gym eval --policy artifacts                        # compare solvers on fresh random workloads
cqlz-gym bench                                          # environment throughput
```

Example (`examples/shop.toml`): the exact solver merges the "customer summary" query into the "orders of
customer" table (3 tables instead of 4, cost 3488 vs 3781), and the printed design re-checks with 0 `cqlc-core` diagnostics. On this tiny example the learned policy does **not** find the merge (it returns the 3781 design); it was trained on larger random workloads and only pays off at scale.

## What is real and what is not

* **Real:** candidate generation, the `cqlc-core` verification (≈ 3.6 µs per call), the exact baseline, the policy
  training, the numbers below.
* **Synthetic:** the **cost model** (seeks, bytes, write amplification, storage; arbitrary units) is not
  calibrated against a real cluster. A policy can exploit its errors; check final designs on real ScyllaDB.
* **Simplified:** one equality-restricted partition, at most one range column, entities are independent, no
  secondary indexes, materialised views or LWT. Candidates are capped (256).
* The "exact" solver is a branch-and-bound over a facility-location formulation, **not** the integer program of
  NoSE (Mior et al., IEEE TKDE 2017). It stands in for it; a faithful NoSE comparison is future work.

## First results (alpha)

Policy: 71,745 parameters (2-layer Transformer encoder over candidate tables), trained 1,000 REINFORCE iterations
(≈ 35 min on one CPU core) on random workloads of 4–30 queries. Evaluation on **fresh random workloads** (different
seed than training), 8 instances per size, `eval --exact-seconds 15`. Values are the mean of `cost / best cost found
by any method on that instance` (1.000 = always best; this is *not* the true optimum when the exact solver times out).

| Workload size | naive | greedy | exact B&B | learned (argmax) | learned (best of 32) | exact solved | exact time | argmax time |
|---|---|---|---|---|---|---|---|---|
| 4–7 queries | 1.002 | 1.004 | 1.000 | 1.000 | 1.000 | 8/8 | < 0.01 s | < 0.01 s |
| 8–16 queries | 1.003 | 1.008 | 1.000 | 1.000 | 1.000 | 8/8 | < 0.01 s | < 0.01 s |
| 25–50 queries | 1.013 | 1.037 | 1.003 | 1.002 | 1.002 | 7/8 | 2.05 s | 0.02 s |
| 80–150 queries | 1.014 | 1.060 | 1.038 | **1.002** | **1.000** | **0/8** | 15.00 s (limit) | 0.31 s |

Reading, honestly:

* Where the exact solver finishes (small and medium) it is trivially best; the policy matches it. The problem is **easy** at
  these sizes, so learning adds nothing there.
* At 80–150 queries the exact branch-and-bound **never finishes** (0/8 within 15 s) and its best incumbent is ~4 % above the
  best design found. The policy's single argmax rollout takes 0.31 s and lands within 0.2 % of the best-known; greedy is 6 % above.
  The best-of-32 variant needs ~10 s because sampling runs on one CPU core.
* Against the pre-registered criteria of the proposal: the policy is far inside the 20 % margin where the exact solver
  works, beats greedy and the timed-out exact solver where it does not, and the environment (observe + step, no network)
  runs at ≈ 227,000 steps/s on one core, above the 10⁴ threshold. Multi-core scaling was **not** measured.

Caveats that matter: the exact baseline is a simple branch-and-bound with a weak bound, not an integer-programming solver
such as NoSE's, and a good ILP solver may do much better at these sizes; the cost model is synthetic and the instance
generator is the same one used for training, so "generalisation" here means to larger sizes, not to real workloads;
`best cost found by any method` is only a reference, not a proof of optimality.

## Research context

See the alpha proposal: <https://github.com/Max-imalgutaussehend/cqlz/blob/main/research/CQL-GYM.md>.
