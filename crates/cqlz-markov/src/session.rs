//! Query-shape session model.

use anyhow::{Result, bail};
use cqlc_core::lexer::{TokenKind, lex};
use cqlc_core::span::FileId;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Normalises a statement to its shape: literals and bind markers become `?`, keywords and identifiers
/// are lower-cased. `SELECT a FROM t WHERE k = 5` and `... k = ?` share a shape.
pub fn shape(cql: &str) -> String {
    let (lexed, _) = lex(FileId(0), cql);
    let mut out: Vec<String> = Vec::new();
    for t in &lexed.tokens {
        let s = match &t.kind {
            TokenKind::Ident(s) | TokenKind::QuotedIdent(s) => s.to_ascii_lowercase(),
            TokenKind::Number(_)
            | TokenKind::String(_)
            | TokenKind::Uuid(_)
            | TokenKind::Blob(_)
            | TokenKind::Bind => "?".into(),
            TokenKind::Semi | TokenKind::Eof => continue,
            TokenKind::LParen => "(".into(),
            TokenKind::RParen => ")".into(),
            TokenKind::LBracket => "[".into(),
            TokenKind::RBracket => "]".into(),
            TokenKind::LBrace => "{".into(),
            TokenKind::RBrace => "}".into(),
            TokenKind::Comma => ",".into(),
            TokenKind::Dot => ".".into(),
            TokenKind::Star => "*".into(),
            TokenKind::Plus => "+".into(),
            TokenKind::Minus => "-".into(),
            TokenKind::Colon => ":".into(),
            TokenKind::Eq => "=".into(),
            TokenKind::Ne => "!=".into(),
            TokenKind::Lt => "<".into(),
            TokenKind::Le => "<=".into(),
            TokenKind::Gt => ">".into(),
            TokenKind::Ge => ">=".into(),
        };
        out.push(s);
    }
    out.join(" ")
}

/// Parses a log of `session_id<TAB>statement` lines into sessions (consecutive lines with the same id;
/// statements are reduced to shapes).
pub fn parse_log(text: &str) -> Vec<Vec<String>> {
    let mut sessions: Vec<Vec<String>> = Vec::new();
    let mut last: Option<&str> = None;
    for line in text.lines() {
        let Some((sid, stmt)) = line.split_once('\t') else {
            continue;
        };
        let sh = shape(stmt.trim().trim_end_matches(';'));
        if sh.is_empty() {
            continue;
        }
        if last != Some(sid) {
            sessions.push(Vec::new());
            last = Some(sid);
        }
        sessions.last_mut().expect("pushed").push(sh);
    }
    sessions
}

/// First-order chain over shapes with an explicit start and end state.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionModel {
    /// Shape strings; index `n` is the virtual `END` state (not stored here, `states.len() == n`).
    pub states: Vec<String>,
    /// `start[i]`: P(first statement is state i).
    pub start: Vec<f64>,
    /// `trans[i][j]`: P(next = j | current = i), `j == n` is END.
    pub trans: Vec<Vec<f64>>,
    pub alpha: f64,
}

impl SessionModel {
    /// Fits by maximum likelihood with additive (Laplace) smoothing `alpha`.
    pub fn fit(sessions: &[Vec<String>], alpha: f64) -> Result<Self> {
        let mut index: HashMap<&str, usize> = HashMap::new();
        let mut states: Vec<String> = Vec::new();
        for s in sessions.iter().flatten() {
            if !index.contains_key(s.as_str()) {
                index.insert(s.as_str(), states.len());
                states.push(s.clone());
            }
        }
        let n = states.len();
        if n == 0 {
            bail!("no statements to fit");
        }
        let mut start = vec![alpha; n];
        let mut trans = vec![vec![alpha; n + 1]; n];
        for sess in sessions {
            let ids: Vec<usize> = sess.iter().map(|s| index[s.as_str()]).collect();
            let Some(&first) = ids.first() else { continue };
            start[first] += 1.0;
            for w in ids.windows(2) {
                trans[w[0]][w[1]] += 1.0;
            }
            trans[*ids.last().expect("non-empty")][n] += 1.0;
        }
        normalise(&mut start);
        trans.iter_mut().for_each(|r| normalise(r));
        Ok(Self {
            states,
            start,
            trans,
            alpha,
        })
    }

    fn id(&self, s: &str) -> Option<usize> {
        self.states.iter().position(|x| x == s)
    }

    /// Mean negative log2-likelihood per statement (incl. the END event) on held-out sessions.
    /// Unknown shapes count as probability `1/(n+1)` (a fixed, documented back-off).
    pub fn bits_per_statement(&self, sessions: &[Vec<String>]) -> f64 {
        let n = self.states.len();
        let unk = 1.0 / (n as f64 + 1.0);
        let (mut bits, mut count) = (0.0, 0usize);
        for sess in sessions {
            let mut prev: Option<usize> = None;
            for s in sess {
                let cur = self.id(s);
                let p = match (prev, cur) {
                    (None, Some(c)) => self.start[c],
                    (Some(p), Some(c)) => self.trans[p][c],
                    _ => unk,
                };
                bits -= p.log2();
                count += 1;
                prev = cur;
            }
            if let Some(p) = prev {
                bits -= self.trans[p][n].log2();
                count += 1;
            }
        }
        bits / count.max(1) as f64
    }

    /// Same statistic for the i.i.d. baseline: next shape drawn from the marginal distribution.
    pub fn unigram_bits(&self, train: &[Vec<String>], test: &[Vec<String>]) -> f64 {
        let n = self.states.len();
        let mut freq = vec![self.alpha; n];
        for s in train.iter().flatten() {
            if let Some(i) = self.id(s) {
                freq[i] += 1.0;
            }
        }
        normalise(&mut freq);
        let unk = 1.0 / (n as f64 + 1.0);
        let (mut bits, mut count) = (0.0, 0usize);
        for s in test.iter().flatten() {
            let p = self.id(s).map_or(unk, |i| freq[i]);
            bits -= p.log2();
            count += 1;
        }
        bits / count.max(1) as f64
    }

    /// Samples one session of shapes.
    pub fn sample(&self, rng: &mut impl Rng, max_len: usize) -> Vec<&str> {
        let n = self.states.len();
        let mut out = Vec::new();
        let mut cur = draw(&self.start, rng);
        while out.len() < max_len {
            out.push(self.states[cur].as_str());
            let next = draw(&self.trans[cur], rng);
            if next == n {
                break;
            }
            cur = next;
        }
        out
    }
}

fn normalise(v: &mut [f64]) {
    let s: f64 = v.iter().sum();
    v.iter_mut().for_each(|x| *x /= s);
}

fn draw(p: &[f64], rng: &mut impl Rng) -> usize {
    let mut u: f64 = rng.random();
    for (i, x) in p.iter().enumerate() {
        if u < *x {
            return i;
        }
        u -= x;
    }
    p.len() - 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn shapes_ignore_literals_and_case() {
        assert_eq!(
            shape("SELECT a FROM t WHERE k = 5;"),
            shape("select a from t where k = ?")
        );
        assert_ne!(
            shape("SELECT a FROM t WHERE k = ?"),
            shape("SELECT a FROM t WHERE k > ?")
        );
    }

    fn synthetic_log(rng: &mut StdRng, sessions: usize) -> String {
        // Ground truth: profile -> orders -> (item | orders)* -> end; a strongly sequential workload.
        let mut s = String::new();
        for i in 0..sessions {
            let seq = [
                "SELECT * FROM users WHERE id = ?",
                "SELECT * FROM orders WHERE user = ?",
            ];
            for q in seq {
                s.push_str(&format!("s{i}\t{q}\n"));
            }
            while rng.random_bool(0.6) {
                let q = if rng.random_bool(0.7) {
                    "SELECT * FROM items WHERE order_id = ?"
                } else {
                    "SELECT * FROM orders WHERE user = ?"
                };
                s.push_str(&format!("s{i}\t{q}\n"));
            }
        }
        s
    }

    #[test]
    fn markov_beats_iid_on_sequential_workloads_and_generation_matches() {
        let mut rng = StdRng::seed_from_u64(3);
        let train = parse_log(&synthetic_log(&mut rng, 2000));
        let test = parse_log(&synthetic_log(&mut rng, 500));
        let m = SessionModel::fit(&train, 0.1).unwrap();
        let markov = m.bits_per_statement(&test);
        let iid = m.unigram_bits(&train, &test);
        assert!(markov < iid - 0.2, "markov {markov} vs iid {iid}");

        // Generated sessions always start with the profile read (a property the i.i.d. model cannot reproduce).
        let starts = (0..500)
            .filter(|_| m.sample(&mut rng, 50)[0].contains("from users"))
            .count();
        assert!(starts > 480, "{starts}");
    }
}
