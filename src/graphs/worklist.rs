use crate::graphs::visited::Visited;

pub trait Worklist<T, V: Visited<T>> {
    fn worklist(self) -> V;
}
