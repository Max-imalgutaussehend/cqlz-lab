# cqlz-markov (alpha)

Markov models for CQL workloads, part of the **cqlz family** research crates.

* **Session chains over query shapes.** Statements are lexed with `cqlc-core` and reduced to shapes (literals
  become `?`). A first-order Markov chain with start/end states is fitted from a `session_id<TAB>statement`
  log, scored by held-out bits per statement against an i.i.d. baseline, and sampled to generate synthetic workloads.
* **Partition growth chain.** Live rows in a partition as a birth-death chain (inserts at rate λ, exponential
  TTL expiry). Exact expected time to a row limit, cross-checked by a Gillespie simulation in the tests.

```sh
cargo install cqlz-markov
cqlz-markov fit sessions.tsv -o model.json        # prints held-out bits/statement: markov vs i.i.d.
cqlz-markov generate model.json --sessions 1000   # synthetic workload
cqlz-markov growth --lambda 50 --limit 1000000                  # no TTL: ~5.6 hours
cqlz-markov growth --lambda 10 --ttl 3600 --limit 100000        # steady state 36 000: effectively never
```

## Status and limits

Alpha. The growth chain models **row counts only**: no tombstones, `gc_grace_seconds`, compaction or bytes.
The tests validate the mathematics (exact vs simulated), not Cassandra's behaviour. The session model is
first-order and needs a real query log to say anything about a real workload.
