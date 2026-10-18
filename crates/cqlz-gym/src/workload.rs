//! Workload description (TOML) and its relational-ish building blocks.

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Attr {
    pub name: String,
    /// CQL type: text, int, bigint, uuid, timestamp, boolean, double, ...
    pub ty: String,
    /// Average stored bytes.
    pub width: f64,
    /// Part of the entity's primary key (needed for row uniqueness in every table).
    #[serde(default)]
    pub key: bool,
    /// Number of distinct values (used by `validate` to generate data). Default: all rows for keys, `sqrt(rows)` otherwise.
    #[serde(default)]
    pub cardinality: Option<f64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Entity {
    pub name: String,
    /// Number of rows.
    pub rows: f64,
    /// Writes (inserts/updates) per second; every table holding the entity pays this.
    pub writes_per_s: f64,
    pub attrs: Vec<Attr>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Query {
    pub name: String,
    pub entity: String,
    pub select: Vec<String>,
    /// Equality-restricted attributes.
    pub eq: Vec<String>,
    /// At most one range-restricted attribute.
    #[serde(default)]
    pub range: Option<String>,
    /// `ORDER BY` attribute (must equal `range` if both are given).
    #[serde(default)]
    pub order: Option<String>,
    /// Queries per second.
    pub weight: f64,
    /// Rows returned per execution.
    pub result_rows: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Workload {
    #[serde(rename = "entity")]
    pub entities: Vec<Entity>,
    #[serde(rename = "query")]
    pub queries: Vec<Query>,
}

impl Workload {
    pub fn from_toml(text: &str) -> Result<Self> {
        let w: Workload = toml::from_str(text).context("parsing workload TOML")?;
        w.validate()?;
        Ok(w)
    }

    pub fn entity(&self, name: &str) -> Option<&Entity> {
        self.entities.iter().find(|e| e.name == name)
    }

    pub fn validate(&self) -> Result<()> {
        for e in &self.entities {
            if !e.attrs.iter().any(|a| a.key) {
                bail!(
                    "entity `{}` needs at least one `key = true` attribute",
                    e.name
                );
            }
        }
        for q in &self.queries {
            let Some(e) = self.entity(&q.entity) else {
                bail!("query `{}`: unknown entity `{}`", q.name, q.entity)
            };
            let has = |c: &String| e.attrs.iter().any(|a| &a.name == c);
            for c in q
                .select
                .iter()
                .chain(&q.eq)
                .chain(q.range.iter())
                .chain(q.order.iter())
            {
                if !has(c) {
                    bail!(
                        "query `{}`: unknown attribute `{c}` of `{}`",
                        q.name,
                        e.name
                    );
                }
            }
            if q.eq.is_empty() {
                bail!(
                    "query `{}`: at least one equality attribute is required (partition key)",
                    q.name
                );
            }
        }
        Ok(())
    }
}
