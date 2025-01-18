use std::f64::consts::PI;

use nalgebra::{Complex, DVector, DVectorView, DVectorViewMut, Dyn};

pub fn fft(x: &DVector<f64>) -> DVector<Complex<f64>> {
    let n = x.len();
    let w = DVector::from_vec(
        (0..(n / 2))
            .map(|i| Complex::new(0.0, -2.0 * i as f64 * PI / n as f64).exp())
            .collect::<Vec<Complex<f64>>>(),
    );

    let mut x_in = DVector::<Complex<f64>>::zeros(n);
    let mut x_out = DVector::<Complex<f64>>::zeros(n);

    for i in 0..n {
        x_in[i] = Complex::new(x[reverse_bits(i, n.ilog2() as usize)], 0.0);
        x_out[i] = Complex::new(x_in[i].re, 0.0);
    }

    dft(
        &mut x_in.rows_mut(0, n / 2),
        &mut x_out.rows_mut(0, n / 2),
        &w.rows_with_step(0, w.len() / 2, 1),
    );
    dft(
        &mut x_in.rows_mut(n / 2, n / 2),
        &mut x_out.rows_mut(n / 2, n / 2),
        &w.rows_with_step(0, w.len() / 2, 1),
    );

    for i in 0..(n / 2) {
        x_out[i] = x_in[i] + w[i] * x_in[(i + n / 2) % n];
        x_out[i + n / 2] = x_in[i] - w[i] * x_in[(i + n / 2) % n];
    }

    x_out
}

fn dft(
    x_out: &mut DVectorViewMut<Complex<f64>>,
    x_in: &mut DVectorViewMut<Complex<f64>>,
    w: &DVectorView<Complex<f64>, Dyn>,
) {
    let n = x_in.nrows();
    if n <= 1 {
        return;
    }

    dft(
        &mut x_in.rows_mut(0, n / 2),
        &mut x_out.rows_mut(0, n / 2),
        &w.rows_with_step(0, w.len() / 2, 1),
    );
    dft(
        &mut x_in.rows_mut(n / 2, n / 2),
        &mut x_out.rows_mut(n / 2, n / 2),
        &w.rows_with_step(0, w.len() / 2, 1),
    );

    for i in 0..(n / 2) {
        x_out[i] = x_in[i] + w[i] * x_in[(i + n / 2) % n];
        x_out[i + n / 2] = x_in[i] - w[i] * x_in[(i + n / 2) % n];
    }
}

fn reverse_bits(input: usize, width: usize) -> usize {
    (0..width).fold(0, |acc, i| (acc << 1) | ((input >> i) & 1))
}

pub fn fftfreq(n: usize, dt: f64) -> DVector<f64> {
    let value = 1.0 / (n as f64 * dt);
    let mut result = DVector::<f64>::zeros(n);
    let mid_n = (n - 1) / 2 + 1;
    for i in 0..mid_n {
        result[i] = value * i as f64;
    }
    for i in 0..(n - mid_n) {
        result[i + mid_n] = value * (i as f64 - (n / 2) as f64);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    use nalgebra::dvector;

    #[test]
    fn test_reverse_bits() {
        assert_eq!(reverse_bits(0, 3), 0);
        assert_eq!(reverse_bits(1, 3), 4);
        assert_eq!(reverse_bits(2, 3), 2);
        assert_eq!(reverse_bits(3, 3), 6);
        assert_eq!(reverse_bits(4, 3), 1);
        assert_eq!(reverse_bits(5, 3), 5);
        assert_eq!(reverse_bits(6, 3), 3);
        assert_eq!(reverse_bits(7, 3), 7);
    }

    #[allow(dead_code)]
    fn naive_fft(x: &DVector<f64>) -> DVector<Complex<f64>> {
        let mut result = DVector::<Complex<f64>>::zeros(x.len());
        for i in 0..x.len() {
            for j in 0..x.len() {
                result[i] += Complex::new(x[j], 0.0)
                    * Complex::new(0.0, -2.0 * j as f64 * PI / x.len() as f64).exp();
            }
        }
        result
    }

    #[test]
    fn test_fft_2() {
        let x = dvector![1.0, 2.0];
        let spectrum = fft(&x);

        let spectrum_real = spectrum.map(|c| c.re);
        let spectrum_imag = spectrum.map(|c| c.im);
        // let expected_spectrum_real = expected_spectrum.map(|c| c.re);
        // let expected_spectrum_imag = expected_spectrum.map(|c| c.im);

        assert_relative_eq!(spectrum_real, dvector![3.0, -1.0]);
        assert_relative_eq!(spectrum_imag, dvector![0.0, 0.0]);
    }

    #[test]
    fn test_fft_4() {
        let x = dvector![1.0, 2.0, 3.0, 4.0];
        let spectrum = fft(&x);

        let spectrum_real = spectrum.map(|c| c.re);
        let spectrum_imag = spectrum.map(|c| c.im);

        assert_relative_eq!(spectrum_real, dvector![10.0, -2.0, -2.0, -2.0]);
        assert_relative_eq!(spectrum_imag, dvector![0.0, 2.0, 0.0, -2.0]);
    }

    #[test]
    fn test_fft_8() {
        let x = dvector![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let spectrum = fft(&x);

        let spectrum_real = spectrum.map(|c| c.re);
        let spectrum_imag = spectrum.map(|c| c.im);

        assert_relative_eq!(
            spectrum_real,
            dvector![36.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0, -4.0],
            epsilon = 1e-14
        );
        assert_relative_eq!(
            spectrum_imag,
            dvector![
                0.0,
                9.65685424949238,
                4.0,
                1.6568542494923797,
                0.0,
                -1.6568542494923797,
                -4.0,
                -9.65685424949238
            ]
        );
    }

    #[test]
    fn test_fftfreq() {
        let n = 5;
        let dt = 1.0;
        let freq = fftfreq(n, dt);
        assert_relative_eq!(freq, dvector![0.0, 0.2, 0.4, -0.4, -0.2]);

        let n = 8;
        let dt = 1.0;
        let freq = fftfreq(n, dt);
        assert_relative_eq!(
            freq,
            dvector![0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]
        );

        let n = 10;
        let dt = 0.2;
        let freq = fftfreq(n, dt);
        assert_relative_eq!(
            freq,
            dvector![0.0, 0.5, 1., 1.5, 2., -2.5, -2., -1.5, -1., -0.5]
        );
    }
}
