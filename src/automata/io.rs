use symbol_table::Symbol;

use crate::{automata::omega::OmegaAutomaton, lattices::set::Set};

/// Input/output label carried by an edge.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct IoLabel {
    /// Consumed input symbol.
    pub input: Symbol,
    /// Produced output symbol.
    pub output: Symbol,
}

impl IoLabel {
    /// Creates a new input/output label.
    #[must_use]
    #[inline]
    pub const fn new(input: Symbol, output: Symbol) -> Self {
        Self { input, output }
    }
}

pub trait IoAutomaton: OmegaAutomaton<Label = IoLabel> {
    #[must_use]
    #[inline]
    fn input_alphabet(&self) -> Set<Symbol> {
        self.alphabet().iter().map(|label| label.input).collect()
    }

    #[must_use]
    #[inline]
    fn output_alphabet(&self) -> Set<Symbol> {
        self.alphabet().iter().map(|label| label.output).collect()
    }
}

impl<T> IoAutomaton for T where T: OmegaAutomaton<Label = IoLabel> {}
