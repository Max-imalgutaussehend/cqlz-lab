# cqlz-gym

**Verifier-in-the-loop design of Cassandra tables.** Part of the **cqlz family** research crates.

Given entities and a query workload (TOML), `cqlz-gym` proposes a set of denormalised tables so that every
query is a clean single-partition read, and prints the `CREATE TABLE` statements plus the queries rewritten
against them. **`cqlc-core` is the exact verifier:** a candidate table may only serve a query if the query
compiles against it with *no* diagnostic, so every proposed design is valid by construction. Solvers: a naive
baseline, greedy, an exact branch-and-bound, an exact integer program (MILP, `good_lp` + `microlp`), and a small Transformer policy (Burn) trained with REINFORCE.

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

* **Real:** candidate generation, the `cqlc-core` verification (≈ 3 µs per call), the exact ILP, the policy training, a cost
  model calibrated on measured latencies of the [cql-hpc-bench](https://huggingface.co/datasets/Max-imumlikelihood/cql-hpc-bench)
  dataset (seek, per-row and per-write cost), and `validate`, which loads designs into a real ScyllaDB and measures them.
* **Simplified:** one equality-restricted partition, at most one range column, entities are independent, no secondary
  indexes, materialised views or LWT. Candidates are capped (256).
* The ILP is a facility-location formulation solved with a pure-Rust MILP solver. It is **not** NoSE itself (Mior et al.,
  IEEE TKDE 2017), but the same kind of model.

## Results

Policy: 71,745 parameters, 1,000 REINFORCE iterations against the calibrated cost model. Evaluation on fresh random
workloads, 8 instances per size, `eval --exact-seconds 15`. Values are the mean of `cost / best cost found by any method`
(1.000 = always best).

| Workload size | naive | greedy | exact B&B | **ILP** | learned (argmax) | learned (best of 32) | B&B solved | ILP solved | ILP time |
|---|---|---|---|---|---|---|---|---|---|
| 4–7 queries | 1.102 | 1.095 | 1.000 | 1.000 | 1.000 | 1.000 | 8/8 | 8/8 | < 0.01 s |
| 8–16 queries | 1.007 | 1.009 | 1.000 | 1.000 | 1.003 | 1.003 | 8/8 | 8/8 | < 0.01 s |
| 25–50 queries | 1.019 | 1.054 | 1.001 | 1.000 | 1.005 | 1.001 | 7/8 | 8/8 | < 0.01 s |
| 80–150 queries | 1.044 | 1.117 | 1.078 | **1.000** | 1.006 | 1.005 | 0/8 | 8/8 | < 0.01 s |

**The honest reading:** the integer program solves every instance to optimality in milliseconds, including the 80–150-query
ones where the hand-written branch-and-bound never finishes. The learned policy is within 0.6 % of it but never better, so
at these sizes **a MILP solver is the right tool and the policy is only a fast approximation**. An earlier version of this
README compared the policy only with the branch-and-bound and showed a win; that comparison was against a weak baseline.
Where learning could still matter (much larger or non-linear cost models) is not shown here.

Environment throughput (`bench`, M3, 4 performance + 4 efficiency cores): 383k steps/s on one thread, 1.40M on four
(91 % efficiency), 2.04M on eight (67 %, efficiency cores). One `cqlc-core` verification takes 3.1 µs.

## Real-database validation

`cqlz-gym validate` loads the naive, greedy and best design into ScyllaDB (Docker, one node, 10,000 rows per entity, 32
clients, 6 s per design) and measures the workload mix.

| Workload | tables naive → best | model-predicted cost change | measured ops/s change | measured weighted latency |
|---|---|---|---|---|
| sensors | 6 → 5 | −23 % | +8.2 % (44,063 → 47,658) | 726 → 671 µs (−7.6 %) |
| shop | 7 → 6 | −5 % | +1.4 % (55,954 → 56,724) | 572 → 564 µs (−1.4 %) |
| killrvideo | 7 → 7 | 0 | within noise (67,594 vs 65,809) | 473 vs 486 µs |

The direction is right and the cost model overstates the gain (−23 % predicted, −7.6 % measured). Run-to-run noise on this
setup is a few percent, so only the sensors result is clearly outside it. Laptop, Docker, single node: ratios, not
absolute numbers.

## Context

Site page: <https://max-imalgutaussehend.github.io/cqlz/cqlz-gym.html>.
