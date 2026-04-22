use symbol_table::Symbol;

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
