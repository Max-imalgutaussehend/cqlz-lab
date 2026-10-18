---
license: apache-2.0
library_name: burn
tags:
  - cassandra
  - scylladb
  - schema-design
  - reinforcement-learning
  - safetensors
---

# cqlz-gym-policy

Small (71,745 parameters) Transformer policy that selects Cassandra table designs, trained by REINFORCE against
`cqlc-core` as an exact verifier. Part of the [cqlz family](https://github.com/Max-imalgutaussehend/cqlz-lab).

* Input: per-candidate features of a design problem (cost, marginal gain, coverage, progress; 8 features).
* Output: logits over candidate tables plus a STOP action; invalid actions are masked by the environment.
* Files: `model.safetensors`, `config.json` (Burn `PolicyConfig`).
* Use: `cargo install cqlz-gym && cqlz-gym solve workload.toml --policy hf`.

## Evaluation (synthetic workloads, cost model calibrated on cql-hpc-bench)

On 80–150-query workloads the policy's argmax rollout (0.16 s) is within 0.6 % of the optimum found by an exact MILP
(which solves every instance in milliseconds); greedy is 11.7 % above and the branch-and-bound incumbent 7.8 % above.
So the policy is a fast approximation, not better than the ILP. On a real ScyllaDB the best design raised throughput
8.2 % (sensors) and 1.4 % (shop) over the naive design. See the repository README for the full table and caveats.
