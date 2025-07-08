use nalgebra::{stack, Complex, DMatrix, DVector};

pub fn polynomial(vec: DVector<Complex<f64>>) -> DVector<Complex<f64>> {
    let mut a = DVector::from_vec(vec![Complex::new(1.0, 0.0)]);
    for x in vec.iter() {
        a = convolve(&a, &DVector::from_vec(vec![Complex::new(1.0, 0.0), -x]));
    }

    a
}

pub fn characteristic_polynomial(matrix: &DMatrix<f64>) -> Option<DVector<f64>> {
    assert_eq!(matrix.nrows(), matrix.ncols(), "Matrix must be square.");

    let complex_eigenvalues = matrix.clone().complex_eigenvalues();
    let mut complex_coeffs = DVector::from_vec(vec![Complex::new(1.0, 0.0)]);

    for complex_eigenvalue in complex_eigenvalues.iter() {
        complex_coeffs = convolve(
            &complex_coeffs,
            &DVector::from_vec(vec![Complex::new(1.0, 0.0), -complex_eigenvalue]),
        );
    }
    let coeffs = DVector::from_vec(complex_coeffs.iter().map(|e| e.re).collect::<Vec<_>>());

    Some(coeffs)
}

pub fn expm(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let n = matrix.nrows();
    let mut result = DMatrix::identity(n, n);
    let mut power = DMatrix::identity(n, n);
    let mut factorial = 1.0;

    for i in 1..=20 {
        power *= matrix;
        factorial *= i as f64;
        result += &power / factorial;
    }

    result
}

pub fn convolve(a: &DVector<Complex<f64>>, b: &DVector<Complex<f64>>) -> DVector<Complex<f64>> {
    let n = a.len();
    let m = b.len();
    let mut result = DVector::from_element(n + m - 1, Complex::new(0.0, 0.0));

    for i in 0..(n + m - 1) {
        let mut sum = Complex::new(0.0, 0.0);
        for k in 0..=i {
            if k < n && (i - k) < m {
                sum += a[k] * b[i - k];
            }
        }
        result[i] = sum;
    }

    result
}

pub fn factorial(n: usize) -> usize {
    if n == 0 {
        1
    } else {
        n * factorial(n - 1)
    }
}

#[allow(dead_code)]
fn correlate(a: &DVector<Complex<f64>>, b: &DVector<Complex<f64>>) -> DVector<Complex<f64>> {
    let mut result = DVector::zeros(a.len() + b.len() - 1);

    let a_padded = stack![DVector::zeros(b.len() - 1); a.clone(); DVector::zeros(b.len() - 1)];
    for i in 0..result.len() {
        for j in 0..b.len() {
            result[i] += a_padded[i + j] * complex_conjugation(&b[j]);
        }
    }

    result
}

#[allow(dead_code)]
fn complex_conjugation(a: &Complex<f64>) -> Complex<f64> {
    Complex::new(a.re, -a.im)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_polynomial() {
        let vec = dvector![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0)
        ];
        let result = polynomial(vec);
        assert_eq!(
            result,
            dvector![
                Complex::new(1.0, 0.0),
                Complex::new(-6.0, 0.0),
                Complex::new(11.0, 0.0),
                Complex::new(-6.0, 0.0)
            ]
        );
    }

    #[test]
    fn test_characteristic_polynomial() {
        let roots = dmatrix![1.0, 0.0; 0.0, 1.0];
        let coeffs = characteristic_polynomial(&roots).unwrap();
        assert_relative_eq!(coeffs, dvector![1.0, -2.0, 1.0]);

        let roots = dmatrix![1.0, 2.0; 3.0, 4.0];
        let coeffs = characteristic_polynomial(&roots).unwrap();
        assert_relative_eq!(coeffs, dvector![1.0, -5.0, -2.0]);

        let roots = dmatrix![-3.0, -1.0; 1.0, 0.0];
        let coeffs = characteristic_polynomial(&roots).unwrap();
        assert_relative_eq!(coeffs, dvector![1.0, 3.0, 1.0]);

        let a = dmatrix![-2.0, -1.0; 1.0, 0.0];
        let b = dmatrix![1.0; 0.0];
        let c = dmatrix![1.0, 2.0];
        assert_relative_eq!(a.clone() - &b * &c, dmatrix![-3.0, -3.0; 1.0, 0.0]);

        assert_relative_eq!(
            characteristic_polynomial(&(a.clone() - (&b * &c))).unwrap(),
            dvector![1.0, 3.0, 3.0]
        );
    }

    #[test]
    fn test_expm() {
        let x = dmatrix![1.0, 1.0; -1.0, 1.0];
        let result = expm(&x);
        assert_relative_eq!(
            result,
            dmatrix![1.46869394, 2.28735529; -2.28735529,  1.46869394],
            epsilon = 1e-8
        );
    }

    #[test]
    fn test_correlate() {
        let a = dvector![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        ];
        let b = dvector![
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.5, 0.0),
        ];
        let real = DVector::from_vec(correlate(&a, &b).iter().map(|e| e.re).collect::<Vec<_>>());
        assert_relative_eq!(real, dvector![0.5, 2.0, 3.5, 3.0, 0.0]);

        let a = dvector![
            Complex::new(1.0, 1.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, -1.0),
        ];
        let b = dvector![
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 0.0),
            Complex::new(0.0, 0.5),
        ];
        let real = DVector::from_vec(correlate(&a, &b).iter().map(|e| e.re).collect::<Vec<_>>());
        let imag = DVector::from_vec(correlate(&a, &b).iter().map(|e| e.im).collect::<Vec<_>>());
        assert_relative_eq!(real, dvector![0.5, 1.0, 1.5, 3.0, 0.0]);
        assert_relative_eq!(imag, dvector![-0.5, 0.0, -1.5, -1.0, 0.0]);

        let a = dvector![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0),
        ];
        let b = dvector![Complex::new(4.0, 0.0), Complex::new(5.0, 0.0)];
        let real = DVector::from_vec(correlate(&a, &b).iter().map(|e| e.re).collect::<Vec<_>>());
        let imag = DVector::from_vec(correlate(&a, &b).iter().map(|e| e.im).collect::<Vec<_>>());
        assert_relative_eq!(real, dvector![5.0, 14.0, 23.0, 12.0]);
        assert_relative_eq!(imag, dvector![0.0, 0.0, 0.0, 0.0]);
    }
}
