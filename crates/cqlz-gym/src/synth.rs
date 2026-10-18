//! Random workload generator for training and evaluation.

use crate::workload::{Attr, Entity, Query, Workload};
use rand::prelude::*;

const NAMES: &[&str] = &[
    "user_id",
    "order_id",
    "tenant",
    "region",
    "device",
    "sensor",
    "day",
    "bucket",
    "ts",
    "created",
    "seq",
    "name",
    "email",
    "status",
    "city",
    "country",
    "price",
    "qty",
    "score",
    "balance",
    "session_id",
    "page",
    "campaign",
    "channel",
    "shard",
    "version",
    "owner",
    "title",
    "body",
    "payload",
    "product_id",
    "category",
    "level",
    "source",
    "account",
    "group_id",
];
const ENTITIES: &[&str] = &[
    "users", "orders", "events", "sensors", "carts", "messages", "invoices", "devices",
];
const TYPES: &[(&str, f64)] = &[
    ("text", 40.0),
    ("int", 4.0),
    ("bigint", 8.0),
    ("uuid", 16.0),
    ("timestamp", 8.0),
    ("double", 8.0),
    ("boolean", 1.0),
];

/// Size parameters of a generated instance.
#[derive(Clone, Copy, Debug)]
pub struct Size {
    pub entities: (usize, usize),
    pub queries: (usize, usize),
}

pub fn workload(rng: &mut impl Rng, size: Size) -> Workload {
    let n_e = rng.random_range(size.entities.0..=size.entities.1);
    let mut names = ENTITIES.to_vec();
    names.shuffle(rng);
    let mut entities = Vec::new();
    for en in names.into_iter().take(n_e) {
        let mut an = NAMES.to_vec();
        an.shuffle(rng);
        let n_a = rng.random_range(6..=10);
        let attrs: Vec<Attr> = an
            .iter()
            .take(n_a)
            .enumerate()
            .map(|(i, n)| {
                let (ty, w) = if i == 0 {
                    ("uuid", 16.0)
                } else {
                    *TYPES.choose(rng).unwrap()
                };
                let width = if ty == "text" {
                    rng.random_range(8.0..200.0)
                } else {
                    w
                };
                Attr {
                    name: (*n).to_string(),
                    ty: ty.to_string(),
                    width,
                    key: i == 0,
                }
            })
            .collect();
        entities.push(Entity {
            name: en.to_string(),
            rows: 10f64.powf(rng.random_range(4.0..8.0)),
            writes_per_s: 10f64.powf(rng.random_range(-1.0..2.5)),
            attrs,
        });
    }
    let n_q = rng.random_range(size.queries.0..=size.queries.1);
    let mut queries = Vec::new();
    for i in 0..n_q {
        let e = entities.choose(rng).unwrap();
        let names: Vec<&String> = e.attrs.iter().map(|a| &a.name).collect();
        let n_eq = if rng.random_bool(0.7) { 1 } else { 2 };
        let mut picks: Vec<&String> = names.clone();
        picks.shuffle(rng);
        let eq: Vec<String> = picks.iter().take(n_eq).map(|s| (*s).clone()).collect();
        let rest: Vec<&&String> = picks.iter().skip(n_eq).collect();
        let range = (rng.random_bool(0.3) && !rest.is_empty()).then(|| (**rest[0]).clone());
        let order = match &range {
            Some(r) => Some(r.clone()),
            None => (rng.random_bool(0.15) && !rest.is_empty()).then(|| (**rest[0]).clone()),
        };
        let n_sel = rng.random_range(1..=4.min(names.len()));
        let mut sel = names.clone();
        sel.shuffle(rng);
        let select: Vec<String> = sel.into_iter().take(n_sel).cloned().collect();
        queries.push(Query {
            name: format!("Q{i}"),
            entity: e.name.clone(),
            select,
            eq,
            range,
            order,
            weight: 10f64.powf(rng.random_range(-0.5..3.0)),
            result_rows: 10f64.powf(rng.random_range(0.0..3.0)),
        });
    }
    Workload { entities, queries }
}
