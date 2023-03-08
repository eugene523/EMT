use rand::Rng;
use crate::math_utils;

pub fn same_size(x: &Vec<f64>, y: &Vec<f64>) -> bool {
    x.len() == y.len()
}

pub fn set_zero(x: &mut Vec<f64>) {
    for i in 0..x.len() { 
        x[i] = 0.0; 
    }
}

pub fn equal_eps(x: &Vec<f64>, y: &Vec<f64>, eps: f64) -> bool {
    assert!(same_size(x, y));
    for i in 0..x.len() {
        if !math_utils::equal_eps(x[i], y[i], eps) {
            return false;
        }
    }
    return true;
}

pub fn equal_zero_eps(x: &Vec<f64>, eps: f64) -> bool {
    for i in 0..x.len() {
        if !math_utils::equal_eps(x[i], 0.0, eps) {
            return false;
        }
    }
    return true;
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

pub fn add_mut(x: &mut Vec<f64>, y: &Vec<f64>) {
    assert!(same_size(x, y));
    let vec_size = x.len();
    for i in 0..vec_size {
        x[i] += y[i];
    }
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

pub fn sub_mut(x: &mut Vec<f64>, y: &Vec<f64>) {
    assert!(same_size(x, y));
    let vec_size = x.len();
    for i in 0..vec_size {
        x[i] -= y[i];
    }
}

pub fn mul(x: &Vec<f64>, coeff: f64) -> Vec<f64> {
    let mut c = vec![0.0; x.len()];
    for i in 0..x.len() {
        c[i] = x[i] * coeff;
    }
    return c;
}

pub fn mul_mut(x: &mut Vec<f64>, coeff: f64) {
    let vec_size = x.len();
    for i in 0..vec_size {
        x[i] *= coeff;
    }
}

pub fn div(x: &Vec<f64>, coeff: f64) -> Vec<f64> {
    assert!(coeff != 0.0);
    return mul(x, 1.0 / coeff);
}

pub fn div_mut(x: &mut Vec<f64>, coeff: f64) {
    assert!(coeff != 0.0);
    mul_mut(x, 1.0 / coeff);
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

pub fn x_plus_by_mut(x: &mut Vec<f64>, b: f64, y: &Vec<f64>) {
    assert!(same_size(x, y));
    for i in 0..x.len() {
        x[i] += b * y[i];
    }
}

pub fn x_plus_by_plus_cz_mut(x: &mut Vec<f64>, b: f64, 
                             y: &Vec<f64>,     c: f64, 
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

pub fn normalize(x: &Vec<f64>) -> Vec<f64> {
    let n = norm(x);
    return div(x, n);
}

pub fn normalize_mut(x: &mut Vec<f64>) {
    let n = norm(x);
    div_mut(x, n);
}

pub fn dot(x: &Vec<f64>, y: &Vec<f64>) -> f64 {
    assert!(same_size(x, y));
    let mut d = 0.0;
    for i in 0..x.len() {
        d += x[i] * y[i];
    }
    return d;
}

pub fn cross3(x: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
    assert!(x.len() == 3);
    assert!(y.len() == 3);
    let mut z = vec![0.0; 3];
    z[0] = x[1] * y[2] - x[2] * y[1];
    z[1] = x[2] * y[0] - x[0] * y[2];
    z[2] = x[0] * y[1] - x[1] * y[0];
    return z;
}

pub fn new_rnd(vect_size: usize) -> Vec<f64> {
    let mut v = vec![0.0; vect_size];
    let mut rng = rand::thread_rng();
    for i in 0..vect_size {
        v[i] = rng.gen::<f64>();
    }
    return v;
}

pub fn new_rnd2(vect_size: usize, nonzero_percent: f64) -> Vec<f64> {
    let mut v = vec![0.0; vect_size];
    let mut rng = rand::thread_rng();
        for i in 0..vect_size {
            if rng.gen_bool(nonzero_percent) { 
                v[i] = rng.gen::<f64>();
            }
        }
        return v;
}

pub fn new_rnd_normalized(vect_size: usize) -> Vec<f64> {
    let mut v = new_rnd(vect_size);
    normalize_mut(&mut v);
    return v;
}

pub fn proj(x: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
    let d1 = dot(x, y);
    let d2 = dot(y, y);
    let d = d1 / d2;
    let x_proj = mul(y, d);
    return x_proj;
}

pub fn orth(x: &Vec<f64>, y: &Vec<f64>) -> Vec<f64> {
    let d1 = dot(x, y);
    let d2 = dot(y, y);
    let d = d1 / d2;
    let mut x_orth = vec![0.0; x.len()];
    for i in 0..x.len() {
        x_orth[i] = x[i] - d * y[i];
    }
    return x_orth;
}

pub fn orth_mut(x: &mut Vec<f64>, y: &Vec<f64>) {
    let d1 = dot(x, y);
    let d2 = dot(y, y);
    let d = d1 / d2;
    for i in 0..x.len() {
        x[i] -= d * y[i];
    }
}

fn orth2_mut(x: &mut Vec<f64>, y: &Vec<f64>, dot_yy: f64) {
    let mut d = dot(x, y);
    d /= dot_yy;
    for i in 0..x.len() {
        x[i] -= d * y[i];
    }
}

pub fn gram_schmidt(vectors: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut orth_vectors: Vec<Vec<f64>> = Vec::new();
    let mut dot_yy: Vec<f64> = Vec::new();

    for i in 0..vectors.len() {
        let mut v = vectors[i].clone();
        for j in 0..i {
            orth2_mut(&mut v, &orth_vectors[j], dot_yy[j]);
        }
        dot_yy.push(dot(&v, &v));
        orth_vectors.push(v);
    }
    return orth_vectors;
}

pub fn gram_schmidt_norm(vectors: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let mut orth_vectors = gram_schmidt(vectors);
    for i in 0..orth_vectors.len() {
        normalize_mut(&mut orth_vectors[i]);
    }
    return orth_vectors;
}

pub fn is_symmetric(x: &Vec<f64>, eps: f64) -> bool {
    let n = x.len();
    let mid = n / 2;
    for i in 0..mid {
        if !math_utils::equal_eps(x[i], x[n - 1 - i], eps) {
            return false;
        }
    }
    return true;
}

pub fn is_antisymmetric(x: &Vec<f64>, eps: f64) -> bool {
    let n = x.len();
    let mid = n / 2;
    for i in 0..mid {
        if !math_utils::equal_eps(x[i], -x[n - 1 - i], eps) {
            return false;
        }
    }
    return true;
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn test_add() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c_test = vec![5.0, 7.0, 9.0];

        let c = add(&a, &b);
        assert!(equal_eps(&c, &c_test, EPS));

        add_mut(&mut a, &b);
        assert!(equal_eps(&a, &c, EPS));
    }

    #[test]
    fn test_sub() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c_test = vec![-3.0, -3.0, -3.0];

        let c = sub(&a, &b);
        assert!(equal_eps(&c, &c_test, EPS));

        sub_mut(&mut a, &b);
        assert!(equal_eps(&a, &c, EPS));
    }

    #[test]
    fn test_mul() {
        let mut a = vec![1.0, 2.0, 3.0];
        let coeff = 3.0;
        let b_test = vec![3.0, 6.0, 9.0];

        let b = mul(&a, coeff);
        assert!(equal_eps(&b, &b_test, EPS));

        mul_mut(&mut a, coeff);
        assert!(equal_eps(&a, &b_test, EPS));
    }

    #[test]
    fn test_div() {
        let mut a = vec![1.0, 2.0, 3.0];
        let coeff = 3.0;
        let b_test = vec![1.0 / 3.0, 2.0 / 3.0, 1.0];

        let b = div(&a, coeff);
        assert!(equal_eps(&b, &b_test, EPS));

        div_mut(&mut a, coeff);
        assert!(equal_eps(&a, &b_test, EPS));
    }

    #[test]
    fn test_ax_by() {
        let x = vec![1.0, 3.0, 7.0];  let x_coeff = 2.0;
        let y = vec![-3.0, 2.0, 5.0]; let y_coeff = 3.0;
        let c_test = vec![-7.0, 12.0, 29.0];

        let c = ax_plus_by(x_coeff, &x, y_coeff, &y);
        assert!(equal_eps(&c, &c_test, EPS));
    }

    #[test]
    fn test_norm() {
        let x = vec![1.0, 2.0, 3.0];
        let n = norm(&x);
        assert!(math_utils::equal_eps(n, 3.74165, 1e-3));
    }

    #[test]
    fn test_normalize() {
        let mut x = vec![1.0, 2.0, 3.0];
        let xn = normalize(&x);
        let xn_test = vec![0.26726, 0.53452, 0.80178];
        assert!(equal_eps(&xn, &xn_test, 1e-3));

        normalize_mut(&mut x);
        assert!(equal_eps(&x, &xn, 1e-3));
    }

    #[test]
    fn test_dot() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        let d = dot(&x, &y);
        let d_test = 32.0;
        assert!(math_utils::equal_eps(d, d_test, EPS));
    }

    #[test]
    fn test_cross3() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![-7.0, 9.0, -15.0];
        let c_test = vec![-57.0, -6.0, 23.0];
        let c = cross3(&x, &y);
        assert!(equal_eps(&c, &c_test, EPS));
    }
}