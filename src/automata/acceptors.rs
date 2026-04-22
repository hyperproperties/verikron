pub trait Acceptor {
    type Summary: ?Sized;

    fn accept(&self, summary: &Self::Summary) -> bool;
}

pub trait FiniteAcceptor: Acceptor {}
pub trait OmegaAcceptor: Acceptor {}
