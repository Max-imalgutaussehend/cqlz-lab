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

# cqlz-gym-policy (alpha)

Small (71,745 parameters) Transformer policy that selects Cassandra table designs, trained by REINFORCE against
`cqlc-core` as an exact verifier. Part of the [cqlz family](https://github.com/Max-imalgutaussehend/cqlz-lab).

* Input: per-candidate features of a design problem (cost, marginal gain, coverage, progress; 8 features).
* Output: logits over candidate tables plus a STOP action; invalid actions are masked by the environment.
* Files: `model.safetensors`, `config.json` (Burn `PolicyConfig`).
* Use: `cargo install cqlz-gym && cqlz-gym solve workload.toml --policy hf`.

## Evaluation (synthetic workloads, synthetic cost model)

On 80–150-query workloads the policy's argmax rollout (0.31 s) is within 0.2 % of the best design found by any method,
while an exact branch-and-bound baseline never finishes in 15 s (incumbent 3.8 % above) and greedy is 6 % above.
See the repository README for the full table and caveats. Alpha: not validated on real clusters.
