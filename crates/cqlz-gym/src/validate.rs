//! Measures designs on a real database: loads every table of a design with data generated from the workload, then runs
//! the weighted query mix plus the writes (every write goes to *all* tables of its entity, which is what denormalisation
//! costs) with concurrent clients and reports throughput and latency.

use crate::problem::{Problem, Table};
use crate::workload::{Attr, Entity, Query};
use anyhow::Result;
use scylla::client::session::Session;
use scylla::client::session_builder::SessionBuilder;
use scylla::errors::TranslationError;
use scylla::policies::address_translator::{AddressTranslator, UntranslatedPeer};
use scylla::statement::prepared::PreparedStatement;
use scylla::value::{CqlTimestamp, CqlValue};
use std::net::{SocketAddr, ToSocketAddrs};
use std::sync::Arc;
use std::time::{Duration, Instant};
use uuid::Uuid;

struct Pin(SocketAddr);

#[async_trait::async_trait]
impl AddressTranslator for Pin {
    async fn translate_address(
        &self,
        _: &UntranslatedPeer,
    ) -> Result<SocketAddr, TranslationError> {
        Ok(self.0)
    }
}

pub async fn connect(node: &str) -> Result<Arc<Session>> {
    let addr = node
        .to_socket_addrs()?
        .next()
        .ok_or_else(|| anyhow::anyhow!("bad node"))?;
    Ok(Arc::new(
        SessionBuilder::new()
            .known_node_addr(addr)
            .address_translator(Arc::new(Pin(addr)))
            .build()
            .await?,
    ))
}

fn cardinality(e: &Entity, a: &Attr, rows: usize) -> usize {
    match a.cardinality {
        Some(c) => (c as usize).clamp(1, rows),
        None if a.key => rows,
        None => ((e.rows.sqrt()) as usize).clamp(1, rows),
    }
}

/// Deterministic value of attribute `a` of row `i`.
fn value(e: &Entity, a: &Attr, i: usize, rows: usize) -> CqlValue {
    let v = if a.key {
        i
    } else {
        i % cardinality(e, a, rows)
    };
    match a.ty.as_str() {
        "text" | "ascii" | "varchar" => {
            let mut s = format!("{}-{v}-", a.name);
            while s.len() < a.width as usize {
                s.push_str("xxxxxxxx");
            }
            s.truncate((a.width as usize).max(s.len().min(24)).max(1));
            CqlValue::Text(s)
        }
        "int" => CqlValue::Int(v as i32),
        "bigint" => CqlValue::BigInt(v as i64),
        "smallint" => CqlValue::SmallInt((v % 30_000) as i16),
        "uuid" => CqlValue::Uuid(Uuid::from_u128(
            ((a.name.len() as u128) << 96) + v as u128 + 1,
        )),
        "timeuuid" => {
            // version-1 layout (version nibble 1, RFC variant) so the server accepts it as a timeuuid
            let raw = ((a.name.len() as u128) << 96) + v as u128 + 1;
            let raw =
                (raw & !(0xf_u128 << 76) & !(0x3_u128 << 62)) | (1_u128 << 76) | (2_u128 << 62);
            CqlValue::Timeuuid(Uuid::from_u128(raw).into())
        }
        "timestamp" => CqlValue::Timestamp(CqlTimestamp(1_700_000_000_000 + i as i64 * 1000)),
        "double" => CqlValue::Double(v as f64 * 0.25),
        "float" => CqlValue::Float(v as f32 * 0.25),
        "boolean" => CqlValue::Boolean(v % 2 == 0),
        _ => CqlValue::Text(format!("{v}")),
    }
}

fn attr<'a>(e: &'a Entity, name: &str) -> &'a Attr {
    e.attrs.iter().find(|a| a.name == name).expect("validated")
}

struct Loaded {
    name: String,
    table: Table,
    entity: Entity,
    insert: PreparedStatement,
}

/// One executable query: statement, how to build parameters from a row index.
struct Op {
    weight: f64,
    kind: OpKind,
}

#[allow(clippy::large_enum_variant)]
enum OpKind {
    Read {
        stmt: PreparedStatement,
        query: Query,
        entity: Entity,
        rows: usize,
    },
    Write {
        targets: Vec<usize>,
        entity: Entity,
        rows: usize,
    },
}

pub struct Outcome {
    pub ops_per_s: f64,
    /// Workload-weighted mean latency in microseconds.
    pub weighted_latency_us: f64,
    pub read_p50_us: f64,
    pub write_p50_us: f64,
    pub rows_loaded: usize,
}

fn percentile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[((v.len() - 1) as f64 * q) as usize]
}

/// Runs one design (`set` of candidate table indices) on `session` in keyspace `ks`.
pub async fn run_design(
    session: &Arc<Session>,
    p: &Problem,
    set: &[usize],
    ks: &str,
    row_cap: usize,
    secs: f64,
    clients: usize,
) -> Result<Outcome> {
    session
        .query_unpaged(format!("DROP KEYSPACE IF EXISTS {ks}"), [])
        .await?;
    session.query_unpaged(format!("CREATE KEYSPACE {ks} WITH replication = {{'class': 'NetworkTopologyStrategy', 'replication_factor': 1}}"), []).await?;
    let rendered = p.render(set);
    let mut loaded = Vec::new();
    let mut names = std::collections::HashMap::new();
    for (k, &t) in set.iter().enumerate() {
        let table = p.tables[t].clone();
        let entity = p.workload.entity(&table.entity).expect("validated").clone();
        let name = format!("{}_{}", table.entity, k + 1);
        let ddl = rendered
            .lines()
            .find(|l| l.starts_with(&format!("CREATE TABLE {name} ")))
            .expect("rendered")
            .replacen(
                &format!("CREATE TABLE {name} "),
                &format!("CREATE TABLE {ks}.{name} "),
                1,
            );
        session.query_unpaged(ddl, []).await?;
        let cols = table.columns.join(", ");
        let marks = vec!["?"; table.columns.len()].join(", ");
        let insert = session
            .prepare(format!("INSERT INTO {ks}.{name} ({cols}) VALUES ({marks})"))
            .await?;
        names.insert(t, loaded.len());
        loaded.push(Loaded {
            name,
            table,
            entity,
            insert,
        });
    }
    // load
    let mut rows_loaded = 0;
    let mut pending = futures::stream::FuturesUnordered::new();
    use futures::StreamExt;
    for l in &loaded {
        let n = (l.entity.rows as usize).min(row_cap).max(1);
        for i in 0..n {
            let vals: Vec<CqlValue> = l
                .table
                .columns
                .iter()
                .map(|c| value(&l.entity, attr(&l.entity, c), i, n))
                .collect();
            let (s, ins) = (session.clone(), l.insert.clone());
            pending.push(async move { s.execute_unpaged(&ins, vals).await });
            rows_loaded += 1;
            if pending.len() >= 256 {
                pending.next().await.transpose()?;
            }
        }
    }
    while let Some(r) = pending.next().await {
        r?;
    }

    // operations: reads of the workload, one write op per entity (all its tables)
    let mut ops: Vec<Op> = Vec::new();
    for (qi, q) in p.workload.queries.iter().enumerate() {
        let best = set
            .iter()
            .filter_map(|&t| p.read[qi][t].map(|c| (c, t)))
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .expect("served")
            .1;
        let li = names[&best];
        let l = &loaded[li];
        let mut conds: Vec<String> = q.eq.iter().map(|c| format!("{c} = ?")).collect();
        if let Some(r) = &q.range {
            conds.push(format!("{r} > ?"));
        }
        let mut cql = format!(
            "SELECT {} FROM {}.{} WHERE {}",
            q.select.join(", "),
            ks,
            l.name,
            conds.join(" AND ")
        );
        if let Some(o) = &q.order {
            cql.push_str(&format!(" ORDER BY {o}"));
        }
        cql.push_str(&format!(" LIMIT {}", (q.result_rows as usize).max(1)));
        let stmt = session.prepare(cql).await?;
        let rows = (l.entity.rows as usize).min(row_cap).max(1);
        ops.push(Op {
            weight: q.weight,
            kind: OpKind::Read {
                stmt,
                query: q.clone(),
                entity: l.entity.clone(),
                rows,
            },
        });
    }
    for e in &p.workload.entities {
        let targets: Vec<usize> = (0..loaded.len())
            .filter(|&i| loaded[i].entity.name == e.name)
            .collect();
        if !targets.is_empty() {
            let rows = (e.rows as usize).min(row_cap).max(1);
            ops.push(Op {
                weight: e.writes_per_s,
                kind: OpKind::Write {
                    targets,
                    entity: e.clone(),
                    rows,
                },
            });
        }
    }
    let loaded = Arc::new(loaded);
    let ops = Arc::new(ops);
    let total_w: f64 = ops.iter().map(|o| o.weight).sum();
    let deadline = Instant::now() + Duration::from_secs_f64(secs);
    let mut tasks = futures::stream::FuturesUnordered::new();
    for c in 0..clients {
        let (session, ops, loaded) = (session.clone(), ops.clone(), loaded.clone());
        tasks.push(tokio::spawn(async move {
            use rand::prelude::*;
            let mut rng = StdRng::seed_from_u64(900 + c as u64);
            let (mut reads, mut writes) = (Vec::new(), Vec::new());
            let mut per_op_sum = vec![(0.0f64, 0u64); ops.len()];
            while Instant::now() < deadline {
                let mut x = rng.random::<f64>() * total_w;
                let mut oi = 0;
                for (i, o) in ops.iter().enumerate() {
                    if x < o.weight {
                        oi = i;
                        break;
                    }
                    x -= o.weight;
                    oi = i;
                }
                let t0 = Instant::now();
                match &ops[oi].kind {
                    OpKind::Read {
                        stmt,
                        query,
                        entity,
                        rows,
                    } => {
                        let i = rng.random_range(0..*rows);
                        let mut vals: Vec<CqlValue> = query
                            .eq
                            .iter()
                            .map(|a| value(entity, attr(entity, a), i, *rows))
                            .collect();
                        if let Some(r) = &query.range {
                            vals.push(value(entity, attr(entity, r), i.saturating_sub(5), *rows));
                        }
                        if session.execute_unpaged(stmt, vals).await.is_err() {
                            break;
                        }
                    }
                    OpKind::Write {
                        targets,
                        entity,
                        rows,
                    } => {
                        let i = rng.random_range(0..*rows);
                        let mut fs = Vec::new();
                        for &ti in targets {
                            let l = &loaded[ti];
                            let vals: Vec<CqlValue> = l
                                .table
                                .columns
                                .iter()
                                .map(|c| value(entity, attr(entity, c), i, *rows))
                                .collect();
                            let (s, ins) = (session.clone(), l.insert.clone());
                            fs.push(async move { s.execute_unpaged(&ins, vals).await.map(|_| ()) });
                        }
                        if futures::future::join_all(fs)
                            .await
                            .into_iter()
                            .any(|r| r.is_err())
                        {
                            break;
                        }
                    }
                }
                let dt = t0.elapsed().as_secs_f64() * 1e6;
                per_op_sum[oi].0 += dt;
                per_op_sum[oi].1 += 1;
                if matches!(ops[oi].kind, OpKind::Read { .. }) {
                    reads.push(dt)
                } else {
                    writes.push(dt)
                }
            }
            (per_op_sum, reads, writes)
        }));
    }
    let n_ops = ops.len();
    let mut sums = vec![(0.0f64, 0u64); n_ops];
    let (mut reads, mut writes) = (Vec::new(), Vec::new());
    while let Some(r) = tasks.next().await {
        let (s, rd, wr) = r?;
        for (a, b) in sums.iter_mut().zip(s) {
            a.0 += b.0;
            a.1 += b.1;
        }
        reads.extend(rd);
        writes.extend(wr);
    }
    let total_ops: u64 = sums.iter().map(|s| s.1).sum();
    let weighted = ops
        .iter()
        .zip(&sums)
        .map(|(o, s)| {
            if s.1 > 0 {
                o.weight * s.0 / s.1 as f64
            } else {
                0.0
            }
        })
        .sum::<f64>()
        / total_w;
    session
        .query_unpaged(format!("DROP KEYSPACE {ks}"), [])
        .await?;
    Ok(Outcome {
        ops_per_s: total_ops as f64 / secs,
        weighted_latency_us: weighted,
        read_p50_us: percentile(&mut reads, 0.5),
        write_p50_us: percentile(&mut writes, 0.5),
        rows_loaded,
    })
}
