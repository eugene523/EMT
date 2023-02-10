#[allow(dead_code)]
pub fn equal_eps(a: f64, b: f64, eps: f64) -> bool {
    if a == b { return true; }

    let a_abs = a.abs();
    let b_abs = b.abs();

    if (a > 0.0 && b > 0.0) || (a < 0.0 && b < 0.0) {
        let min_val = f64::min(a_abs, b_abs);
        let max_val = f64::max(a_abs, b_abs);
        let d = (max_val - min_val) / min_val;
        if d < eps { return true; }
    }
    return (a_abs < (eps / 2.0)) && (b_abs < (eps / 2.0));
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_nearly_equal() {
        use super::*;

        let eps = 1e-2;
        let (mut a, mut b) = (0.0, 0.0);
        assert!(equal_eps(a, b, eps));

        a =  7.0;   b =  7.001; assert!(equal_eps(a, b, eps));
        a = -5.0;   b = -5.001; assert!(equal_eps(a, b, eps));
        a = -0.003; b =  0.004; assert!(equal_eps(a, b, eps));
        a = -100.0; b =  100.0; assert!(!equal_eps(a, b, eps));
        a =  100.0; b =  102.0; assert!(!equal_eps(a, b, eps));
        a = -100.0; b = -102.0; assert!(!equal_eps(a, b, eps));

    }
}
