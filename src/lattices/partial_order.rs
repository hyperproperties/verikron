pub trait PartialOrder: PartialEq + PartialOrd {}

impl<T: PartialEq + PartialOrd + ?Sized> PartialOrder for T {}
