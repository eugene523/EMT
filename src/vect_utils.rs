use crate::math_utils;

pub fn same_size(x: &Vec<f64>, y: &Vec<f64>) -> bool {
    x.len() == y.len()
}

pub fn add(x: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
    assert!(same_size(x, y));
    let vec_size = x.len();
    let mut c = vec![0.0; vec_size];
    for i in 0..vec_size {
        c[i] = x[i] + y[i];
    }
    return c;
}

pub fn sub(x: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
    assert!(same_size(x, y));
    let vec_size = x.len();
    let mut c = vec![0.0; vec_size];
    for i in 0..vec_size {
        c[i] = x[i] - y[i];
    }
    return c;
}

pub fn mul(x: &Vec<f64>, coeff: f64) -> Vec<f64> {
    let mut c = vec![0.0; x.len()];
    for i in 0..x.len() {
        c[i] = x[i] * coeff;
    }
    return c;
}

//pub fn mul_mut()

pub fn div(x: &Vec<f64>, coeff: f64) -> Vec<f64> {
    assert!(coeff != 0.0);
    return mul(x, 1.0 / coeff);
}

pub fn ax_plus_by(a: f64, x: &Vec<f64>, 
                  b: f64, y: &Vec<f64>) -> Vec<f64>
{
    assert!(same_size(x, y));
    let mut c = vec![0.0; x.len()];
    for i in 0..x.len() {
        c[i] = a * x[i] + b * y[i];
    }
    return c;
}

pub fn x_plus_by_mut(x: &mut Vec<f64>, 
                     b: f64, 
                     y: &Vec<f64>) 
{
    assert!(same_size(x, y));
    for i in 0..x.len() {
        x[i] += b * y[i];
    }
}

pub fn x_plus_by_plus_cz_mut(x: &mut Vec<f64>,
                             b: f64,
                             y: &Vec<f64>,
                             c: f64,
                             z: &Vec<f64>) 
{
    assert!(same_size(x, y));
    assert!(same_size(y, z));
    for i in 0..x.len() {
        x[i] += b * y[i] + c * z[i];
    }
}

pub fn norm(x: &Vec<f64>) -> f64 {
    let mut n = 0.0;
    for i in 0..x.len() {
        n += x[i] * x[i];
    }
    return n.sqrt();
}

