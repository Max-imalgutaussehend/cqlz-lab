//! Candidate tables, the `cqlc-core` verifier and the (explicitly synthetic) cost model.

use crate::workload::{Entity, Query, Workload};
use cqlc_core::{Input, compile};
use std::collections::{BTreeSet, HashMap};

/// Cost model constants; the unit is "one point read" per second of traffic.
///
/// [`CostModel::default`] is **calibrated on measurements**: on ScyllaDB (Apple M3, Docker, the `cql-hpc-bench`
/// dataset of the `cql-ai` repository) a partition read that returns `n` rows of `w` bytes costs about
/// `1 + n * (0.0047 + w / 65,946)` point reads, fitted on 168 measured reads of 100+ rows. Storage cost stays an
/// arbitrary weight. [`CostModel::legacy`] is the uncalibrated model of the first alpha.
#[derive(Clone, Copy, Debug)]
pub struct CostModel {
    /// Fixed cost of one partition read.
    pub seek: f64,
    /// Cost of materialising one returned row, independent of its width.
    pub per_row: f64,
    /// Bytes that cost as much as one seek.
    pub unit: f64,
    /// Storage cost per GB.
    pub storage_per_gb: f64,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            seek: 1.0,
            per_row: 0.0047,
            unit: 65_946.0,
            storage_per_gb: 50.0,
        }
    }
}

impl CostModel {
    /// The synthetic model of the first alpha (a seek equals 4 KiB, no per-row cost).
    pub fn legacy() -> Self {
        Self {
            seek: 1.0,
            per_row: 0.0,
            unit: 4096.0,
            storage_per_gb: 50.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Table {
    pub entity: String,
    pub partition: Vec<String>,
    pub clustering: Vec<String>,
    /// All columns (keys included).
    pub columns: Vec<String>,
}

impl Table {
    pub fn name(&self, i: usize) -> String {
        format!("{}_t{}", self.entity, i)
    }
}

/// A fully verified design problem: which candidate can serve which query, and what everything costs.
pub struct Problem {
    pub workload: Workload,
    pub tables: Vec<Table>,
    /// `read[q][t]`: cost per second of query `q` on table `t`, `None` if `cqlc-core` rejects it.
    pub read: Vec<Vec<Option<f64>>>,
    /// Write + storage cost of opening table `t`.
    pub open: Vec<f64>,
    /// Number of `cqlc-core` compilations spent on verification.
    pub verifier_calls: usize,
}

fn union_sorted(a: &[String], b: &[String]) -> Vec<String> {
    let mut s: BTreeSet<String> = a.iter().cloned().collect();
    s.extend(b.iter().cloned());
    s.into_iter().collect()
}

fn canonical(w: &Workload, q: &Query) -> Table {
    let e = w.entity(&q.entity).expect("validated");
    let mut partition = q.eq.clone();
    partition.sort();
    partition.dedup();
    let mut clustering: Vec<String> = Vec::new();
    for c in q.range.iter().chain(q.order.iter()) {
        if !clustering.contains(c) && !partition.contains(c) {
            clustering.push(c.clone());
        }
    }
    for a in e.attrs.iter().filter(|a| a.key) {
        if !clustering.contains(&a.name) && !partition.contains(&a.name) {
            clustering.push(a.name.clone());
        }
    }
    let mut columns: Vec<String> = q.select.clone();
    columns.extend(q.eq.clone());
    columns.extend(q.range.clone());
    columns.extend(q.order.clone());
    columns.extend(partition.iter().cloned());
    columns.extend(clustering.iter().cloned());
    columns.sort();
    columns.dedup();
    Table {
        entity: q.entity.clone(),
        partition,
        clustering,
        columns,
    }
}

/// Canonical tables, merges of tables with identical keys, and key-compatible variants.
pub fn candidates(w: &Workload, cap: usize) -> Vec<Table> {
    let canon: Vec<Table> = w.queries.iter().map(|q| canonical(w, q)).collect();
    let mut out: Vec<Table> = Vec::new();
    let mut seen: HashMap<Table, ()> = HashMap::new();
    let mut push = |t: Table, out: &mut Vec<Table>| {
        if seen.insert(t.clone(), ()).is_none() {
            out.push(t);
        }
    };
    for t in &canon {
        push(t.clone(), &mut out);
    }
    // Group by (entity, partition, clustering): merging unions the columns.
    let mut groups: HashMap<(String, Vec<String>, Vec<String>), Vec<usize>> = HashMap::new();
    for (i, t) in canon.iter().enumerate() {
        groups
            .entry((t.entity.clone(), t.partition.clone(), t.clustering.clone()))
            .or_default()
            .push(i);
    }
    let mut keys: Vec<_> = groups.keys().cloned().collect();
    keys.sort();
    for k in keys {
        let idx = &groups[&k];
        let m = idx.len();
        let subsets: Vec<Vec<usize>> = if m <= 4 {
            (1u32..(1 << m))
                .map(|mask| {
                    (0..m)
                        .filter(|b| mask >> b & 1 == 1)
                        .map(|b| idx[b])
                        .collect()
                })
                .collect()
        } else {
            let mut v: Vec<Vec<usize>> = idx.iter().map(|&i| vec![i]).collect();
            v.push(idx.clone());
            for a in 0..m {
                for b in a + 1..m {
                    v.push(vec![idx[a], idx[b]]);
                }
            }
            v
        };
        for s in subsets {
            let mut cols: Vec<String> = Vec::new();
            for &i in &s {
                cols = union_sorted(&cols, &canon[i].columns);
            }
            push(
                Table {
                    entity: k.0.clone(),
                    partition: k.1.clone(),
                    clustering: k.2.clone(),
                    columns: cols,
                },
                &mut out,
            );
        }
    }
    // Key-compatible variants: the clustering of one query's table under another's partition.
    for a in &canon {
        for b in &canon {
            if a.entity == b.entity
                && a.partition == b.partition
                && a.clustering != b.clustering
                && out.len() < cap
            {
                push(
                    Table {
                        columns: union_sorted(&a.columns, &b.columns),
                        clustering: a.clustering.clone(),
                        ..a.clone()
                    },
                    &mut out,
                );
            }
        }
    }
    out.truncate(cap);
    out
}

fn ddl(e: &Entity, t: &Table, name: &str) -> String {
    let cols: Vec<String> = t
        .columns
        .iter()
        .map(|c| {
            let a = e.attrs.iter().find(|a| &a.name == c).expect("validated");
            format!("{} {}", a.name, a.ty)
        })
        .collect();
    let mut key = format!("({})", t.partition.join(", "));
    for c in &t.clustering {
        key.push_str(&format!(", {c}"));
    }
    format!(
        "CREATE TABLE {name} ({}, PRIMARY KEY ({key}));",
        cols.join(", ")
    )
}

fn query_cql(q: &Query, table: &str) -> String {
    let mut s = format!(
        "-- name: Q :many\nSELECT {} FROM {table} WHERE ",
        q.select.join(", ")
    );
    let mut conds: Vec<String> = q.eq.iter().map(|c| format!("{c} = ?")).collect();
    if let Some(r) = &q.range {
        conds.push(format!("{r} > ?"));
    }
    s.push_str(&conds.join(" AND "));
    if let Some(o) = &q.order {
        s.push_str(&format!(" ORDER BY {o}"));
    }
    s.push(';');
    s
}

/// The exact verifier: does `cqlc-core` accept `q` on `t` with *no* diagnostics (errors or warnings)?
pub fn serves(e: &Entity, t: &Table, q: &Query) -> bool {
    if t.entity != q.entity {
        return false;
    }
    let needs = q
        .select
        .iter()
        .chain(&q.eq)
        .chain(q.range.iter())
        .chain(q.order.iter());
    if needs.into_iter().any(|c| !t.columns.contains(c)) {
        return false;
    }
    let c = compile(
        &[Input::new("s.cql", ddl(e, t, "t"))],
        &[Input::new("q.cql", query_cql(q, "t"))],
    );
    c.diagnostics.is_empty()
}

pub fn width(e: &Entity, t: &Table) -> f64 {
    t.columns
        .iter()
        .map(|c| {
            e.attrs
                .iter()
                .find(|a| &a.name == c)
                .map_or(0.0, |a| a.width)
        })
        .sum()
}

impl Problem {
    pub fn build(workload: Workload, cost: CostModel, cap: usize) -> Problem {
        let tables = candidates(&workload, cap);
        let mut calls = 0;
        let mut read = vec![vec![None; tables.len()]; workload.queries.len()];
        for (qi, q) in workload.queries.iter().enumerate() {
            let e = workload.entity(&q.entity).expect("validated");
            for (ti, t) in tables.iter().enumerate() {
                if t.entity != q.entity {
                    continue;
                }
                calls += 1;
                if serves(e, t, q) {
                    read[qi][ti] = Some(
                        q.weight
                            * (cost.seek
                                + q.result_rows * (cost.per_row + width(e, t) / cost.unit)),
                    );
                }
            }
        }
        let open = tables
            .iter()
            .map(|t| {
                let e = workload.entity(&t.entity).expect("validated");
                let w = width(e, t);
                e.writes_per_s * (1.0 + w / cost.unit) + e.rows * w / 1e9 * cost.storage_per_gb
            })
            .collect();
        Problem {
            workload,
            tables,
            read,
            open,
            verifier_calls: calls,
        }
    }

    pub fn n_queries(&self) -> usize {
        self.workload.queries.len()
    }

    /// Queries that no candidate can serve (must be empty for a solvable problem).
    pub fn unservable(&self) -> Vec<&str> {
        (0..self.n_queries())
            .filter(|&q| self.read[q].iter().all(Option::is_none))
            .map(|q| self.workload.queries[q].name.as_str())
            .collect()
    }

    /// Total cost of a table set; `None` if some query is not served.
    pub fn cost(&self, set: &[usize]) -> Option<f64> {
        let mut total: f64 = set.iter().map(|&t| self.open[t]).sum();
        for q in 0..self.n_queries() {
            total += set
                .iter()
                .filter_map(|&t| self.read[q][t])
                .fold(None, |m: Option<f64>, c| Some(m.map_or(c, |x| x.min(c))))?;
        }
        Some(total)
    }

    /// `CREATE TABLE` statements and rewritten queries for a chosen set, with the cheapest server per query.
    pub fn render(&self, set: &[usize]) -> String {
        let mut out = String::new();
        let mut names = HashMap::new();
        for (k, &t) in set.iter().enumerate() {
            let tb = &self.tables[t];
            let name = format!("{}_{}", tb.entity, k + 1);
            let e = self.workload.entity(&tb.entity).expect("validated");
            out.push_str(&format!("{}\n", ddl(e, tb, &name)));
            names.insert(t, name);
        }
        out.push('\n');
        for (qi, q) in self.workload.queries.iter().enumerate() {
            let best = set
                .iter()
                .filter_map(|&t| self.read[qi][t].map(|c| (c, t)))
                .min_by(|a, b| a.0.total_cmp(&b.0));
            if let Some((_, t)) = best {
                out.push_str(
                    &query_cql(q, &names[&t])
                        .replace("-- name: Q", &format!("-- name: {}", q.name)),
                );
                out.push('\n');
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solve::{exact, greedy, naive};
    use crate::workload::Workload;
    use std::time::Duration;

    const SHOP: &str = include_str!("../examples/shop.toml");

    #[test]
    fn verifier_accepts_aligned_and_rejects_misaligned_tables() {
        let w = Workload::from_toml(SHOP).unwrap();
        let p = Problem::build(w, CostModel::default(), 64);
        assert!(p.unservable().is_empty());
        let e = p.workload.entity("orders").unwrap();
        let by_customer = &p.workload.queries[1]; // eq customer_id, range placed_at
        let by_order = Table {
            entity: "orders".into(),
            partition: vec!["order_id".into()],
            clustering: vec![],
            columns: vec![
                "order_id".into(),
                "customer_id".into(),
                "placed_at".into(),
                "status".into(),
                "total".into(),
            ],
        };
        // partition key order_id cannot serve a query restricted on customer_id
        assert!(!serves(e, &by_order, by_customer));
        let ok = Table {
            entity: "orders".into(),
            partition: vec!["customer_id".into()],
            clustering: vec!["placed_at".into(), "order_id".into()],
            columns: vec![
                "order_id".into(),
                "customer_id".into(),
                "placed_at".into(),
                "status".into(),
                "total".into(),
            ],
        };
        assert!(serves(e, &ok, by_customer));
    }

    #[test]
    fn solver_ordering_and_verified_output() {
        let w = Workload::from_toml(SHOP).unwrap();
        let p = Problem::build(w, CostModel::default(), 64);
        let (n, g, x) = (naive(&p), greedy(&p), exact(&p, Duration::from_secs(5)));
        assert!(x.optimal);
        assert!(
            x.cost <= g.cost + 1e-9 && g.cost <= n.cost + 1e-9,
            "{} {} {}",
            x.cost,
            g.cost,
            n.cost
        );
        assert!(
            x.cost < n.cost,
            "merging a summary query into the customer table must pay off"
        );

        // The rendered design compiles against cqlc-core without any diagnostic.
        let design = p.render(&x.set);
        let (schema, queries): (Vec<_>, Vec<_>) =
            design.lines().partition(|l| l.starts_with("CREATE"));
        let c = cqlc_core::compile(
            &[cqlc_core::Input::new("s.cql", schema.join("\n"))],
            &[cqlc_core::Input::new("q.cql", queries.join("\n"))],
        );
        assert!(
            c.diagnostics.is_empty(),
            "{:?}",
            c.diagnostics
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn random_workloads_are_solvable_and_exact_is_never_worse_than_greedy() {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(5);
        let mut checked = 0;
        while checked < 15 {
            let w = crate::synth::workload(
                &mut rng,
                crate::synth::Size {
                    entities: (1, 2),
                    queries: (3, 6),
                },
            );
            let p = Problem::build(w, CostModel::default(), 64);
            if !p.unservable().is_empty() {
                continue;
            }
            let (g, x) = (greedy(&p), exact(&p, Duration::from_secs(5)));
            assert!(x.cost <= g.cost + 1e-9);
            assert!(p.cost(&x.set).is_some());
            checked += 1;
        }
    }
}
