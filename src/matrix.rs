use crate::math_utils;
use rand::Rng;
use std::fmt;

#[allow(dead_code)]
#[derive(Clone)]
pub struct Matrix {
    nrows: usize,
    ncols: usize,
    vals: Vec<f64>
}

pub struct LuDec {
    lu_matrix: Matrix,
    permutation_vec: Vec<usize>
}

#[allow(dead_code)]
impl Matrix {
    pub fn new(nrows: usize, ncols: usize) -> Matrix {
        let mut vals: Vec<f64> = Vec::new();
        vals.resize(nrows * ncols, 0.0);
        Matrix { nrows, ncols, vals }
    }

    pub fn new_square(mat_size: usize) -> Matrix {
        let mut vals: Vec<f64> = Vec::new();
        vals.resize(mat_size * mat_size, 0.0);
        Matrix { nrows: mat_size, ncols: mat_size, vals }
    }

    fn same_size(&self, another: &Matrix) -> bool {
        self.nrows == another.nrows && self.ncols == another.ncols
    }

    fn is_square(&self) -> bool { self.nrows == self.ncols }

    pub fn set(&mut self, row: usize, col: usize, val: f64) {
        debug_assert!(row < self.nrows);
        debug_assert!(col < self.ncols);
        let i = row * self.ncols + col;
        self.vals[i] = val;
    }

    pub fn get(&self, row: usize, col: usize) -> f64 {
        debug_assert!(row < self.nrows);
        debug_assert!(col < self.ncols);
        let i = row * self.ncols + col;
        self.vals[i]
    }

    ///////////////////////////////////////////////////////////////////////////

    pub fn set_to_zero(&mut self) {
        for i in self.vals.iter_mut() { *i = 0.0; }
    }

    pub fn is_zero(&self) -> bool {
        for v in self.vals.iter() {
            if *v != 0.0 { return false; }
        }
        return true;
    }

    pub fn identity(mat_size: usize) -> Matrix {
        let mut m = Matrix::new(mat_size, mat_size);
        for i in 0..mat_size {
            m.set(i, i, 1.0);
        }
        return m;
    }

    pub fn set_to_identity(&mut self) {
        debug_assert!(self.nrows == self.ncols);
        self.set_to_zero();
        for i in 0..self.nrows {
            self.set(i, i, 1.0);
        }
    }

    pub fn kron(&self, n_blocks: usize) -> Matrix {
        let k_nrows = n_blocks * self.nrows;
        let k_ncols = n_blocks * self.ncols;
        let mut k = Matrix::new(k_nrows, k_ncols);
        for i in 0..n_blocks {
            let rd = i * self.nrows;
            let cd = i * self.ncols;
            for r in 0..self.nrows {
                for c in 0..self.ncols {
                    let val = self.get(r, c);
                    k.set(rd + r, cd + c, val);
                }
            }
        }
        return k;
    }

    pub fn rnd(nrows: usize, ncols: usize, nonzero_percent: f64) -> Matrix {
        assert!(nonzero_percent >= 0.0 && nonzero_percent <= 1.0);
        let mut m = Matrix::new(nrows, ncols);
        if nonzero_percent == 0.0 { return m; }
        let nvals = nrows * ncols;
        let mut rng = rand::thread_rng();
        for i in 0..nvals {
            if rng.gen_bool(nonzero_percent) { 
                m.vals[i] = rng.gen::<f64>();
            }
        }
        return m;
    }

    pub fn rnd_square(mat_size: usize, nonzero_percent: f64) -> Matrix {
        Matrix::rnd(mat_size, mat_size, nonzero_percent)
    }

    pub fn rnd_square_tria(mat_size: usize, nonzero_percent: f64) -> Matrix {
        assert!(nonzero_percent >= 0.0 && nonzero_percent <= 1.0);
        let mut m = Matrix::new_square(mat_size);
        if nonzero_percent == 0.0 { return m; }
        let mut rng = rand::thread_rng();
        for r in 0..mat_size {
            for c in 0..=r {
                if rng.gen_bool(nonzero_percent) {
                    m.set(r, c, rng.gen::<f64>());
                }
            }
        }
        return m;
    }

    pub fn rnd_posdef(mat_size: usize, nonzero_percent: f64) -> Matrix {
        let a = Matrix::rnd_square(mat_size, nonzero_percent);
        let a_tr = a.transpose();
        let mut p = a.mul(&a_tr);
        for diag in 0..p.nrows {
            let index = p.ncols * diag + diag;
            p.vals[index] += 1.0;
        }
        return p;
    }

    ///////////////////////////////////////////////////////////////////////////

    pub fn set_row_zero(&mut self, r: usize) {
        assert!(r < self.nrows);
        let row_begin = r * self.ncols;
        for c in 0..self.ncols {
            self.vals[row_begin + c] = 0.0;
        }
    }

    pub fn set_col_zero(&mut self, c: usize) {
        assert!(c < self.ncols);
        for r in 0..self.nrows {
            self.vals[r * self.ncols + c] = 0.0;
        }
    }

    ///////////////////////////////////////////////////////////////////////////

    pub fn swap_rows(&mut self, r1: usize, r2: usize) {
        assert!(r1 < self.nrows);
        assert!(r2 < self.nrows);
        for c in 0..self.ncols {
            let i1 = r1 * self.ncols + c;
            let i2 = r2 * self.ncols + c;
            self.vals.swap(i1, i2);
        }
    }

    pub fn swap_cols(&mut self, c1: usize, c2: usize) {
        assert!(c1 < self.ncols);
        assert!(c2 < self.ncols);
        for r in 0..self.nrows {
            let i1 = r * self.ncols + c1;
            let i2 = r * self.ncols + c2;
            self.vals.swap(i1, i2);
        }
    }

    ///////////////////////////////////////////////////////////////////////////

    pub fn set_block(&mut self, 
                     block_size: usize, 
                     block_row:  usize, 
                     block_col:  usize, 
                     block:      &Matrix)
    {
        assert!(self.nrows % block_size == 0);
        assert!(self.ncols % block_size == 0);

        assert!(block_row < (self.nrows / block_size));
        assert!(block_col < (self.ncols / block_size));

        assert!(block.nrows == block_size);
        assert!(block.ncols == block_size);

        for r in 0..block_size {
            for c in 0..block_size {
                let r_index = block_row * block_size + r;
                let c_index = block_col * block_size + c;
                let val = block.get(r, c);
                self.set(r_index, c_index, val);
            }
        }
    }

    pub fn get_block(&self,
                     block_size: usize, 
                     block_row:  usize, 
                     block_col:  usize) -> Matrix
    {
        assert!(self.nrows % block_size == 0);
        assert!(self.ncols % block_size == 0);

        assert!(block_row < (self.nrows / block_size));
        assert!(block_col < (self.ncols / block_size));

        let mut block = Matrix::new_square(block_size);

        for r in 0..block_size {
            for c in 0..block_size {
                let r_index = block_row * block_size + r;
                let c_index = block_col * block_size + c;
                let val = self.get(r_index, c_index);
                block.set(r, c, val);
            }
        }
        return block;
    }

    pub fn add_block(&mut self, 
                     block_size: usize, 
                     block_row:  usize, 
                     block_col:  usize, 
                     block:      &Matrix)
    {
        assert!(self.nrows % block_size == 0);
        assert!(self.ncols % block_size == 0);

        assert!(block_row < (self.nrows / block_size));
        assert!(block_col < (self.ncols / block_size));

        assert!(block.nrows == block_size);
        assert!(block.ncols == block_size);

        for r in 0..block_size {
            for c in 0..block_size {
                let r_index = block_row * block_size + r;
                let c_index = block_col * block_size + c;
                let v1 = self.get(r_index, c_index);
                let v2 = block.get(r, c);
                self.set(r_index, c_index, v1 + v2);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////////////

    pub fn transpose(&self) -> Matrix {
        let mut t = Matrix::new(self.ncols, self.nrows);
        for c in 0..self.ncols {
            for r in 0..self.nrows {
                let value = self.get(r, c);
                t.set(c, r, value);
            }
        }
        return t;
    }

    pub fn add(&self, another: &Matrix) -> Matrix {
        debug_assert!(self.nrows == another.nrows);
        debug_assert!(self.ncols == another.ncols);
        let mut m = Matrix::new(self.nrows, self.ncols);
        for i in 0..(self.nrows * self.ncols) {
            m.vals[i] = self.vals[i] + another.vals[i];
        }
        return m;
    }

    pub fn add_mod(&mut self, another: &Matrix) {
        debug_assert!(self.nrows == another.nrows);
        debug_assert!(self.ncols == another.ncols);
        for i in 0..(self.nrows * self.ncols) {
            self.vals[i] += another.vals[i];
        }
    }

    pub fn sub(&self, another: &Matrix) -> Matrix {
        debug_assert!(self.nrows == another.nrows);
        debug_assert!(self.ncols == another.ncols);
        let mut m = Matrix::new(self.nrows, self.ncols);
        for i in 0..(self.nrows * self.ncols) {
            m.vals[i] = self.vals[i] - another.vals[i];
        }
        return m;
    }

    pub fn sub_mod(&mut self, another: &Matrix) {
        debug_assert!(self.nrows == another.nrows);
        debug_assert!(self.ncols == another.ncols);
        for i in 0..(self.nrows * self.ncols) {
            self.vals[i] -= another.vals[i];
        }
    }

    pub fn mul(&self, another: &Matrix) -> Matrix {
        debug_assert!(self.ncols == another.nrows);
        let mut m = Matrix::new(self.nrows, another.ncols);
        for r in 0..self.nrows {
            for c in 0..another.ncols {
                let mut sum = 0.0;
                for k in 0..self.ncols {
                    sum += self.get(r, k) * another.get(k, c);
                }
                m.set(r, c, sum);
            }
        }
        return m;
    }

    pub fn mul_by_coeff(&self, coeff: f64) -> Matrix {
        let mut m = Matrix::new(self.nrows, self.ncols);
        for i in 0..(self.nrows * self.ncols) {
            m.vals[i] = self.vals[i] * coeff;
        }
        return m;
    }

    pub fn mul_by_coeff_mod(&mut self, coeff: f64) {
        for i in 0..(self.nrows * self.ncols) {
            self.vals[i] *= coeff;
        }
    }

    pub fn div_by_coeff(&self, coeff: f64) -> Matrix {
        self.mul_by_coeff(1.0 / coeff)
    }

    pub fn div_by_coeff_mod(&mut self, coeff: f64) {
        self.mul_by_coeff_mod(1.0 / coeff);
    }

    pub fn invert(&self) -> Option<Matrix> {
        assert!(self.nrows == self.ncols);

        let mut vals: Vec<f64> = self.vals.clone();
        let mut e = Matrix::identity(self.nrows);

        // Проходим по диагонали
        for diag in 0..self.nrows {

            // Элемент на диагонали равен нулю -> менеяем строки
            if vals[diag * self.ncols + diag] == 0.0 {

                // Если последний элемент на диагонали равен 0, то матрица вырождена.
                if diag == (self.nrows - 1) { return None; }

                // Ищем строку с ненулевым начальным элементом
                for nz in (diag + 1)..self.nrows 
                {
                    if vals[nz * self.ncols + diag] != 0.0 
                    {
                        for c in 0..self.ncols 
                        {
                            let d_index = diag * self.ncols + c;
                            let n_index = nz * self.ncols + c;
                            vals.swap(d_index, n_index);
                            e.vals.swap(d_index, n_index);
                        }
                        break;
                    }
                    if nz == (self.nrows - 1) { return None; }
                }
            }
            
            // Элемент диагонали не равен нулю -> 
            // ищем строки по всей матрице!!! с ненулевым элементом.
            for r in 0..self.nrows {
                if r == diag { continue; }
                if vals[r * self.ncols + diag] == 0.0 { continue; }

                let coeff = vals[r * self.ncols + diag] / vals[diag * self.ncols + diag];
                for j in 0..self.ncols {
                    vals[r * self.ncols + j] -= coeff * vals[diag * self.ncols + j];
                    e.vals[r * self.ncols + j] -= coeff * e.vals[diag * self.ncols + j];
                }
            }
        }

        // Делим элементы бывшей единичной матрицы на диагональные элементы матрицы.
        for n in 0..self.nrows {
            for m in 0..self.ncols {
                e.vals[n * self.ncols + m] /= vals[n * self.ncols + n];
            }
        }
        Some(e)
    }

    pub fn det(&self) -> f64 {
        assert!(self.nrows == self.ncols);

        let mut vals: Vec<f64> = self.vals.clone();
        let mut det_coeff = 1;

        // Проходим по диагонали
        for diag in 0..self.nrows {

            // Элемент на диагонали равен нулю -> менеяем строки
            if vals[diag * self.ncols + diag] == 0.0 {

                // Если последний элемент на диагонали равен 0, то матрица вырождена.
                if diag == (self.nrows - 1) { return 0.0; }

                // Ищем строку с ненулевым начальным элементом
                for nz in (diag + 1)..self.nrows 
                {
                    if vals[nz * self.ncols + diag] != 0.0 
                    {
                        for c in 0..self.ncols 
                        {
                            let d_index = diag * self.ncols + c;
                            let n_index = nz * self.ncols + c;
                            vals.swap(d_index, n_index);
                        }
                        break;
                    }
                    if nz == (self.nrows - 1) { return 0.0; }
                }
                det_coeff *= -1;
            }
            
            // Элемент диагонали не равен нулю -> 
            // ищем строки по всей матрице!!! с ненулевым элементом.
            for r in (diag + 1)..self.nrows {
                if vals[r * self.ncols + diag] == 0.0 { continue; }
                let coeff = vals[r * self.ncols + diag] / vals[diag * self.ncols + diag];
                for j in 0..self.ncols {
                    vals[r * self.ncols + j] -= coeff * vals[diag * self.ncols + j];
                }
            }
        }

        let mut det = 1.0;
        for diag in 0..self.nrows {
            det *= vals[diag * self.ncols + diag];
        }
        det *= det_coeff as f64;
        return det;
    }

    pub fn sum(&self) -> f64 {
        let mut s = 0.0;
        for i in 0..self.vals.len() {
            s += self.vals[i];
        }
        return s;
    }

    pub fn equal_eps(&self, another: &Matrix, eps: f64) -> bool {
        if !self.same_size(another) { return false; }
        let nvals = self.nrows * self.ncols;
        for i in 0..nvals {
            let a = self.vals[i];
            let b = another.vals[i];
            if !math_utils::equal_eps(a, b, eps) { return false; }
        }
        return true;
    }

    ///////////////////////////////////////////////////////////////////////////

    fn find_max_row_below_diag(&self, diag: usize) -> Option<usize> {
        let mut max_val = 0.0;
        let mut row = 0;
        for r in diag..self.nrows {
            let val = self.get(r, diag).abs();
            if val > max_val {
                max_val = val;
                row = r;
            }
        }
        if max_val != 0.0 { 
            return Some(row); 
        }
        return None;
    }

    pub fn ludec(&self) -> Option<LuDec> {
        assert!(self.is_square());
        let mut m = self.clone();
        let mut permutation_vec: Vec<usize> = Vec::new();
        permutation_vec.resize(m.nrows, 0);
        for i in 0..m.nrows {
            permutation_vec[i] = i;
        }

        for diag in 0..m.nrows
        {
            // Находим номер ряда, содержащего максимальный элемент под диагональным.
            let row_max = match self.find_max_row_below_diag(diag) {
                Some(row_max) => row_max,

                // Если он не найден, то матрица вырождена.
                None => { return None; }
            };

            // Если номер ряда, в котором находится максимальный элемент
            // не равен номеру текущему номеру ряда с диагональным элементом,
            // то обмениваем ряды.
            if row_max != diag {
                m.swap_rows(diag, row_max);
                permutation_vec.swap(diag, row_max);
            }

            // Вычисляем дополнение Шура
            let diag_val = m.get(diag, diag);
            for r in (diag + 1)..m.nrows {
                let mut v = m.get(r, diag);
                v /= diag_val;
                m.set(r, diag, v);
                for c in (diag + 1)..m.ncols {
                    let mut s = m.get(r, c);
                    let w = m.get(diag, c);
                    s -= v * w;
                    m.set(r, c, s);
                }
            }
        }
        Some(LuDec { lu_matrix: m, permutation_vec })
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "Matrix({}, {})", self.nrows, self.ncols)?;
        for r in 0..self.nrows {
            for c in 0..self.ncols {
                write!(f, "{} ", self.get(r, c))?;
            }
            write!(f, "\n")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn test_set_get() {
        let mut m = Matrix::new(3, 3);
        let val = 3.5;
        m.set(1, 2, val);
        assert_eq!(m.get(1, 2), val);

    }

    #[test]
    fn test_equal_eps() {
        let mut a = Matrix::new_square(2);
        a.set(0, 0, 1.5);
        a.set(0, 1, 2.0);
        a.set(1, 0, -5.157);
        a.set(1, 1, 10.517);

        let mut b = Matrix::new_square(2);
        b.set(0, 0, 1.5);
        b.set(0, 1, 2.0);
        b.set(1, 0, -5.157);
        b.set(1, 1, 10.517);

        assert!(a.equal_eps(&b, EPS));

        let mut c = Matrix::new_square(2);
        c.set(0, 0, 1.5);
        c.set(0, 1, 2.1); // Это значение отличается.
        c.set(1, 0, -5.157);
        c.set(1, 1, 10.517);

        assert!(!a.equal_eps(&c, EPS));
    }

    #[test]
    fn test_set_to_zero() {
        let mut m = Matrix::rnd(5, 5, 1.0);
        m.set_to_zero();
        assert!(m.sum() == 0.0);
        assert!(m.is_zero());
    }

    #[test]
    fn test_set_row_zero() {
        let mut m = Matrix::rnd(3, 4, 1.0);
        for r in 0..m.nrows {
            m.set_row_zero(r);
        }
        assert!(m.is_zero());
    }

    #[test]
    fn test_set_col_zero() {
        let mut m = Matrix::rnd(3, 4, 1.0);
        for c in 0..m.ncols {
            m.set_col_zero(c);
        }
        assert!(m.is_zero());
    }

    #[test]
    fn test_identity() {
        let m = Matrix::identity(5);
        assert!(m.sum() == 5.0);
    }

    #[test]
    fn test_sum() {
        let mut m = Matrix::new_square(2);
        m.set(0, 0, 1.5);
        m.set(0, 1, 2.5);
        m.set(1, 0, 3.6);
        m.set(1, 1, 10.7);
        print!("{}", m.sum());
        assert!(math_utils::equal_eps(m.sum(), 18.3, EPS));
    }

    #[test]
    fn test_kron() {
        let n_blocks = 3;
        let mut a = Matrix::new_square(2);
        a.set(0, 0, 1.0);
        a.set(0, 0, 2.0);
        a.set(1, 0, 3.0);
        a.set(1, 1, 4.0);
        let a_sum = a.sum();
        let k = a.kron(3);
        let k_sum = k.sum();
        assert!(math_utils::equal_eps(n_blocks as f64 * a_sum, k_sum, EPS));
    }

    #[test]
    fn test_transpose() {
        let m = Matrix::rnd(3, 5, 1.0);
        let m_tr = m.transpose();
        assert!(!m.equal_eps(&m_tr, EPS));
        assert!(math_utils::equal_eps(m.sum(), m_tr.sum(), EPS));

        let m_tr2 = m_tr.transpose();
        assert!(m_tr2.equal_eps(&m, EPS));
        assert!(math_utils::equal_eps(m.sum(), m_tr2.sum(), EPS));
    }

    #[test]
    fn test_swap_rows() {
        let m_original = Matrix::rnd(3, 3, 1.0);
        let mut m = m_original.clone();
        m.swap_rows(0, 1);
        m.swap_rows(1, 2);
        m.swap_rows(0, 2);
        m.swap_rows(1, 2);
        assert!(m.equal_eps(&m_original, EPS));
    }

    #[test]
    fn test_swap_cols() {
        let m_original = Matrix::rnd(3, 3, 1.0);
        let mut m = m_original.clone();
        m.swap_cols(0, 1);
        m.swap_cols(1, 2);
        m.swap_cols(0, 2);
        m.swap_cols(1, 2);
        assert!(m.equal_eps(&m_original, EPS));
    }

    #[test]
    fn test_get_set_block() {
        let b_size = 3;
        let b_rows = 3;
        let b_cols = 3;

        let mut a = Matrix::new(b_rows * b_size, b_cols * b_size);
        let mut blocks: Vec<Matrix> = Vec::new();
        for r in 0..b_rows {
            for c in 0..b_cols {
                let b = Matrix::rnd(b_size, b_size, 1.0);
                a.set_block(b_size, r, c, &b);
                blocks.push(b);
            }
        }

        for r in 0..b_rows {
            for c in 0..b_cols {
                let b = a.get_block(b_size, r, c);
                let index = r * b_cols + c;
                let b_test = &blocks[index];
                assert!(b.equal_eps(b_test, EPS));
            }
        }
    }

    #[test]
    fn test_add_block() {
        let mut a = Matrix::new(4, 4);
        let mut b = Matrix::new(2, 2);
        b.set(0, 0, 1.0);
        b.set(0, 1, 2.0);
        b.set(1, 0, 3.0);
        b.set(1, 1, 4.0);
        a.add_block(2, 1, 1, &b);

        let mut c = Matrix::new(4, 4);
        c.set(2, 2, 1.0);
        c.set(2, 3, 2.0);
        c.set(3, 2, 3.0);
        c.set(3, 3, 4.0);

        assert!(a.equal_eps(&c, EPS));
    }

    #[test]
    fn test_add() {
        let nrows = 4;
        let ncols = 5;

        let mut a = Matrix::rnd(nrows, ncols, 1.0);
        let mut a_sum = a.sum();

        let b = Matrix::rnd(nrows, ncols, 1.0);
        let b_sum = b.sum();

        let c = a.add(&b);
        let c_sum = c.sum();
        assert!(math_utils::equal_eps(a_sum + b_sum, c_sum, EPS));

        a.add_mod(&b);
        a_sum = a.sum();
        assert!(math_utils::equal_eps(a_sum, c_sum, EPS));
    }

    #[test]
    fn test_sub() {
        let nrows = 4;
        let ncols = 5;

        let mut a = Matrix::rnd(nrows, ncols, 1.0);
        let mut a_sum = a.sum();

        let b = Matrix::rnd(nrows, ncols, 1.0);
        let b_sum = b.sum();

        let c = a.sub(&b);
        let c_sum = c.sum();
        assert!(math_utils::equal_eps(a_sum - b_sum, c_sum, EPS));

        a.sub_mod(&b);
        a_sum = a.sum();
        assert!(math_utils::equal_eps(a_sum, c_sum, EPS));
    }

    #[test]
    fn test_mul_by_coeff() {
        let mut a = Matrix::rnd(3, 5, 1.0);
        let mut a_sum = a.sum();

        let coeff = 2.77;
        let b = a.mul_by_coeff(coeff);
        let b_sum = b.sum();
        assert!(math_utils::equal_eps(a_sum * coeff, b_sum, EPS));

        a.mul_by_coeff_mod(coeff);
        a_sum = a.sum();
        assert!(math_utils::equal_eps(a_sum, b_sum, EPS));
    }

    #[test]
    fn test_div_by_coeff() {
        let mut a = Matrix::rnd(3, 5, 1.0);
        let mut a_sum = a.sum();

        let coeff = 2.77;
        let b = a.div_by_coeff(coeff);
        let b_sum = b.sum();
        assert!(math_utils::equal_eps(a_sum / coeff, b_sum, EPS));

        a.div_by_coeff_mod(coeff);
        a_sum = a.sum();
        assert!(math_utils::equal_eps(a_sum, b_sum, EPS));
    }

    #[test]
    fn test_mul() {
        let a = Matrix::rnd(3, 5, 1.0);
        let b = Matrix::rnd(5, 5, 1.0);
        let c = Matrix::rnd(5, 5, 1.0);

        // Testing property: (A * B) * C = A * (B * C)
        let ab = a.mul(&b);
        let left = ab.mul(&c);
        let bc = b.mul(&c);
        let right = a.mul(&bc);
        assert!(left.equal_eps(&right, EPS));

        // Testing property: A * (B + C) = A * B + A * C
        let bc = b.add(&c);
        let left = a.mul(&bc);
        let ab = a.mul(&b);
        let ac = a.mul(&c);
        let right = ab.add(&ac);
        assert!(left.equal_eps(&right, EPS));

        // Testing property: (A * B)_tr = B_tr * A_tr
        let ab = a.mul(&b);
        let left = ab.transpose();
        let a_tr = a.transpose();
        let b_tr = b.transpose();
        let right = b_tr.mul(&a_tr);
        assert!(left.equal_eps(&right, EPS));
    }

    #[test]
    fn test_invert() {
        let eps = 1e-8;
        let mat_size = 10;
        let a = Matrix::rnd(mat_size, mat_size, 0.8);
        let b = Matrix::rnd(mat_size, mat_size, 0.8);
        let i = Matrix::identity(mat_size);

        let ainv = a.invert().unwrap();
        let binv = b.invert().unwrap();
        let iinv = i.invert().unwrap();

        // Testing property: I_inv = I
        assert!(i.equal_eps(&iinv, eps));

        // Testing property: A * A_inv = I.
        let mut left = a.mul(&ainv);
        assert!(left.equal_eps(&i, eps));

        // Testing property: A_inv * A = I.
        left = ainv.mul(&a);
        assert!(left.equal_eps(&i, eps));

        // Testing property: (A * B)_inv = B_inv * A_inv.
        let ab = a.mul(&b);
        left = ab.invert().unwrap();
        let mut right = binv.mul(&ainv);
        assert!(left.equal_eps(&right, eps));

        // Testing property: (A_tr)_inv = (A_inv)_tr.
        left = a.transpose().invert().unwrap();
        right = a.invert().unwrap().transpose();
        assert!(left.equal_eps(&right, eps));

        // Testing property: (k * A)_inv = (1 / k) * A_inv.
        let k = 2.37;
        left = a.mul_by_coeff(k).invert().unwrap();
        right = a.invert().unwrap().div_by_coeff(k);
        assert!(left.equal_eps(&right, eps));

        // Testing impossibility of inverting singular matrix.
        let mut zero_column_matrix = Matrix::rnd_square(mat_size, 1.0);
        zero_column_matrix.set_col_zero(1);
        assert!(zero_column_matrix.invert().is_none());

        let mut zero_row_matrix = Matrix::rnd_square(mat_size, 1.0);
        zero_row_matrix.set_row_zero(1);
        assert!(zero_row_matrix.invert().is_none());
    }

    #[test]
    fn test_det() {
        let mat_size = 5;
        let eps = 1e-8;

        // Testing property: det(I) = 1.
        let i = Matrix::identity(mat_size);
        let i_det = i.det();
        assert!(math_utils::equal_eps(i_det, 1.0, eps));

        let a = Matrix::rnd_square(mat_size, 1.0);

        // Testing property: det(A_tr) = det(A)
        let mut left = a.transpose().det();
        let mut right = a.det();
        assert!(math_utils::equal_eps(left, right, eps));

        // Testing property: det(A_inv) = 1 / det(A);
        let a_inv = a.invert().unwrap();
        left = a_inv.det();
        right = 1.0 / a.det();
        assert!(math_utils::equal_eps(left, right, eps));

        // Testing that determinant of a singular matrix equals zero.
        let mut zero_column_matrix = Matrix::rnd_square(mat_size, 1.0);
        zero_column_matrix.set_col_zero(2);
        assert!(zero_column_matrix.det() == 0.0);

        let mut zero_row_matrix = Matrix::rnd_square(mat_size, 1.0);
        zero_row_matrix.set_row_zero(2);
        assert!(zero_row_matrix.det() == 0.0);
    }
}