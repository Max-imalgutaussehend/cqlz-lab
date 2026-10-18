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
customer" table (3 tables instead of 4, cost 3488 vs 3781), and the printed design re-checks with 0 `cqlc-core` diagnostics.

## What is real and what is not

* **Real:** candidate generation, the `cqlc-core` verification (≈ 3.6 µs per call), the exact baseline, the policy
  training, the numbers below.
* **Synthetic:** the **cost model** (seeks, bytes, write amplification, storage; arbitrary units) is not
  calibrated against a real cluster. A policy can exploit its errors; check final designs on real ScyllaDB.
* **Simplified:** one equality-restricted partition, at most one range column, entities are independent, no
  secondary indexes, materialised views or LWT. Candidates are capped (256).
* The "exact" solver is a branch-and-bound over a facility-location formulation, **not** the integer program of
  NoSE (Mior et al., IEEE TKDE 2017). It stands in for it; a faithful NoSE comparison is future work.

RESULTS_PLACEHOLDER

## Research context

See the alpha proposal: <https://github.com/Max-imalgutaussehend/cqlz/blob/main/research/CQL-GYM.md>.
