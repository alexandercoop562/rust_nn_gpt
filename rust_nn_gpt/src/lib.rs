#![deny(clippy::unwrap_used)]
#![deny(clippy::expect_used)]
#![deny(clippy::panic)]
#![deny(clippy::implicit_return)]
#![allow(clippy::needless_return)]
#![deny(clippy::assigning_clones)]
#![deny(clippy::implicit_clone)]
#![deny(unused_must_use)]

pub use tch::{Device, Tensor};

pub mod dataset;
pub mod tokenizer;
pub mod transformer;
