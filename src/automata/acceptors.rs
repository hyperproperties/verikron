use std::hash::Hash;

use crate::automata::trace::Summary;

pub trait Acceptor<S: Eq + Hash> {
    fn accepts(&self, summary: &Summary<S>) -> bool;
}
