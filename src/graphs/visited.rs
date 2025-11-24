pub trait Visited<V>: Default {
    fn visit(&mut self, value: V) -> bool;

    fn is_visited(&self, value: &V) -> bool;
}
