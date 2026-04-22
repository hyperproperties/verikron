use crate::{automata::io::IoLabel, lattices::set::Set};

pub type Alphabet<L> = Set<L>;
pub type IoAlphabet = Alphabet<IoLabel>;
