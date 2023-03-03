#![allow(dead_code)]
mod math_utils;
mod vect_utils;
mod matrix;

fn main() {
    let m = matrix::Matrix::rnd_square_tria(3, 1.0);
    print!("{}", m);
}
