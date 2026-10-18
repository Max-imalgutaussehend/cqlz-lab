# cqlz-lab

Home of **cqlz-gym** in the **cqlz family** ([site](https://max-imalgutaussehend.github.io/cqlz/cqlz-gym.html)).
Everything here is early, small and honest about its limits.

| Crate | What |
|---|---|
| [`cqlz-gym`](crates/cqlz-gym) | Verifier-in-the-loop design of Cassandra tables: `cqlc-core` checks every candidate, a small Burn policy picks the table set |

Related repositories: [`cqlc`](https://github.com/Max-imalgutaussehend/cqlc), [`cqlx`](https://github.com/Max-imalgutaussehend/cqlx),
[`cql-ai`](https://github.com/Max-imalgutaussehend/cql-ai).

```sh
cargo test --workspace
cargo run --release -p cqlz-gym -- solve crates/cqlz-gym/examples/shop.toml
```

Apache-2.0
