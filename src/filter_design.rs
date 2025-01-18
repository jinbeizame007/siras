use std::f64::consts::PI;

use nalgebra::{dvector, stack, Complex, DVector};

use crate::math::{factorial, polynomial};
use crate::transfer_function::ContinuousTransferFunction;

pub fn butter(order: usize, cutoff_frequency: f64) -> ContinuousTransferFunction {
    let num = dvector![cutoff_frequency.powf(order as f64)];

    let thetas: Vec<f64> = (1..=order)
        .map(|k| PI * (2 * k + order - 1) as f64 / (2 * order) as f64)
        .collect();
    let poles: Vec<Complex<f64>> = thetas
        .iter()
        .map(|theta| cutoff_frequency * Complex::new(theta.cos(), theta.sin()))
        .filter(|p| p.re <= 0.0)
        .collect();
    let den_complex = polynomial(DVector::from_vec(poles));
    let den = DVector::from_vec(den_complex.iter().map(|e| e.re).collect::<Vec<_>>());

    ContinuousTransferFunction::new(num, den)
}

pub fn bessel(order: usize, cutoff_frequency: f64) -> ContinuousTransferFunction {
    let den_bessel = reverse_bessel_polynomial(order);
    let den_cutoff = DVector::from_vec(
        (0..=order)
            .rev()
            .map(|k| (1.0 / cutoff_frequency).powf(k as f64))
            .collect::<Vec<_>>(),
    );
    let den = den_bessel.component_mul(&den_cutoff);
    let num = dvector![den_bessel[den_bessel.len() - 1]];

    ContinuousTransferFunction::new(num, den)
}

fn reverse_bessel_polynomial(order: usize) -> DVector<f64> {
    let mut coeffs = DVector::zeros(order + 1);
    coeffs[0] = 1.0;
    for k in 0..order {
        coeffs[order - k] = (factorial(2 * order - k)
            / (usize::pow(2, (order - k) as u32) * factorial(k) * factorial(order - k)))
            as f64;
    }

    coeffs
}

pub fn chebyshev1(
    order: usize,
    cutoff_frequency: f64,
    ripple_db: f64,
) -> ContinuousTransferFunction {
    let ripple = f64::sqrt(10.0_f64.powf(ripple_db / 10.0) - 1.0);
    let num =
        dvector![cutoff_frequency.powf(order as f64) / (2.0_f64.powf(order as f64 - 1.0) * ripple)];

    let mut poles: DVector<Complex<f64>> = DVector::zeros(order);
    for k in 1..=order {
        let theta = (PI / 2.0) * (2.0 * k as f64 - 1.0) / order as f64;
        poles[k - 1] = Complex::new(
            -1.0 * (cutoff_frequency
                * ((1.0 / order as f64) * (1.0 / ripple).asinh()).sinh()
                * theta.sin())
            .abs(),
            cutoff_frequency * ((1.0 / order as f64) * (1.0 / ripple).asinh()).cosh() * theta.cos(),
        );
    }
    let den = DVector::from_vec(polynomial(poles).iter().map(|e| e.re).collect::<Vec<_>>());

    ContinuousTransferFunction::new(num, den)
}

pub fn chebyshev1_polynomial(order: usize) -> DVector<f64> {
    if order == 0 {
        return dvector![1.0];
    } else if order == 1 {
        return dvector![1.0, 0.0];
    } else {
        let mut polynomials: Vec<DVector<f64>> = vec![dvector![1.0], dvector![1.0, 0.0]];
        for k in 2..=order {
            polynomials.push(
                stack![2.0 * polynomials[k - 1].clone(); dvector![0.0]]
                    - stack![dvector![0.0, 0.0]; polynomials[k - 2].clone()],
            );
        }

        return polynomials[order].clone();
    }
}

pub fn chebyshev2(
    order: usize,
    cutoff_frequency: f64,
    ripple_db: f64,
) -> ContinuousTransferFunction {
    let ripple = 1.0 / f64::sqrt(10.0_f64.powf(ripple_db / 10.0) - 1.0);

    let mut poles_num_vec: Vec<Complex<f64>> = vec![];
    let mut poles_den: DVector<Complex<f64>> = DVector::zeros(order);
    for k in 1..=order {
        let theta = (PI / 2.0) * (2.0 * k as f64 - 1.0) / order as f64;
        if (2 * k - 1) != order {
            poles_num_vec.push(Complex::new(0.0, -cutoff_frequency / theta.cos()));
        }
        poles_den[k - 1] = 1.0
            / Complex::new(
                (-1.0 / cutoff_frequency)
                    * (((1.0 / order as f64) * (1.0 / ripple).asinh()).sinh() * theta.sin()).abs(),
                (1.0 / cutoff_frequency)
                    * ((1.0 / order as f64) * (1.0 / ripple).asinh()).cosh()
                    * theta.cos(),
            );
    }
    let poles_num = DVector::from_vec(poles_num_vec);
    let mut num = DVector::from_vec(
        polynomial(poles_num)
            .iter()
            .map(|e| e.re)
            .collect::<Vec<_>>(),
    );
    let den = DVector::from_vec(
        polynomial(poles_den)
            .iter()
            .map(|e| e.re)
            .collect::<Vec<_>>(),
    );

    // Normalize the numerator
    num *= den[den.len() - 1] / num[num.len() - 1];

    ContinuousTransferFunction::new(num, den)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_butter() {
        // 1st order: (s + 1)
        let tf = butter(1, 1.0);
        assert_eq!(tf.num, dvector![1.0]);
        assert_relative_eq!(tf.den, dvector![1.0, 1.0]);

        // 2nd order: (s^2 + sqrt(2)s + 1)
        let tf = butter(2, 1.0);
        assert_eq!(tf.num, dvector![1.0]);
        assert_relative_eq!(tf.den, dvector![1.0, f64::sqrt(2.0), 1.0]);

        // 3rd order: (s + 1)(s^2 + s + 1)
        let tf = butter(3, 1.0);
        assert_eq!(tf.num, dvector![1.0]);
        assert_relative_eq!(tf.den, dvector![1.0, 2.0, 2.0, 1.0]);

        // 4th order: (s^2 + sqrt(2 - sqrt(2))s + 1)(s^2 + sqrt(2 + sqrt(2))s + 1)
        let tf = butter(4, 1.0);
        assert_eq!(tf.num, dvector![1.0]);
        assert_relative_eq!(
            tf.den,
            dvector![
                1.0,
                (2.0 + f64::sqrt(2.0)).sqrt() + (2.0 - f64::sqrt(2.0)).sqrt(),
                2.0 + (2.0 + f64::sqrt(2.0)).sqrt() * (2.0 - f64::sqrt(2.0)).sqrt(),
                (2.0 + f64::sqrt(2.0)).sqrt() + (2.0 - f64::sqrt(2.0)).sqrt(),
                1.0
            ],
            epsilon = 1e-14
        );
    }

    #[test]
    fn test_bessel() {
        let tf = bessel(1, 1.0);
        assert_eq!(tf.num, dvector![1.0]);
        assert_relative_eq!(tf.den, dvector![1.0, 1.0]);

        let tf = bessel(2, 1.0);
        assert_eq!(tf.num, dvector![3.0]);
        assert_relative_eq!(tf.den, dvector![1.0, 3.0, 3.0]);

        let tf = bessel(3, 1.0);
        assert_eq!(tf.num, dvector![15.0]);
        assert_relative_eq!(tf.den, dvector![1.0, 6.0, 15.0, 15.0]);

        let tf = bessel(4, 1.0);
        assert_eq!(tf.num, dvector![105.0]);
        assert_relative_eq!(tf.den, dvector![1.0, 10.0, 45.0, 105.0, 105.0]);

        let tf = bessel(5, 1.0);
        assert_eq!(tf.num, dvector![945.0]);
        assert_relative_eq!(tf.den, dvector![1.0, 15.0, 105.0, 420.0, 945.0, 945.0]);
    }

    #[test]
    fn test_chebyshev1_polynomial() {
        let poly = chebyshev1_polynomial(0);
        assert_eq!(poly, dvector![1.0]);

        let poly = chebyshev1_polynomial(1);
        assert_eq!(poly, dvector![1.0, 0.0]);

        let poly = chebyshev1_polynomial(2);
        assert_eq!(poly, dvector![2.0, 0.0, -1.0]);

        let poly = chebyshev1_polynomial(3);
        assert_eq!(poly, dvector![4.0, 0.0, -3.0, 0.0]);

        let poly = chebyshev1_polynomial(4);
        assert_eq!(poly, dvector![8.0, 0.0, -8.0, 0.0, 1.0]);
    }

    #[test]
    fn test_chebyshev1() {
        let tf = chebyshev1(1, 100.0, 1.0);
        assert_relative_eq!(tf.num, dvector![196.52267283602717]);
        assert_relative_eq!(tf.den, dvector![1.0, 196.52267283602717]);

        let tf = chebyshev1(2, 100.0, 1.0);
        assert_relative_eq!(tf.num, dvector![9826.133641801356]);
        assert_relative_eq!(
            tf.den,
            dvector![1.0, 109.77343285639276, 11025.103280538484]
        );

        let tf = chebyshev1(3, 100.0, 1.0);
        assert_relative_eq!(tf.num, dvector![491306.6820900678]);
        assert_relative_eq!(
            tf.den,
            dvector![1.0, 98.8341209884761, 12384.091735782364, 491306.6820900678],
            epsilon = 1e-9
        );

        let tf = chebyshev1(4, 90.0, 0.1);
        assert_relative_eq!(tf.num, dvector![53736256.63180374], epsilon = 1e-7);
        assert_relative_eq!(
            tf.den,
            dvector![
                1.0,
                162.33952536509088,
                21277.06074788149,
                1476589.879470203,
                54358493.15756986,
            ]
        );

        let tf = chebyshev1(1, 100.0, 3.0);
        assert_relative_eq!(tf.num, dvector![100.23772930076005]);
        assert_relative_eq!(tf.den, dvector![1.0, 100.23772930076005]);

        let tf = chebyshev1(2, 100.0, 3.0);
        assert_relative_eq!(tf.num, dvector![5011.886465038001], epsilon = 1e-11);
        assert_relative_eq!(tf.den, dvector![1.0, 64.48996513028668, 7079.477801252795]);

        let tf = chebyshev1(3, 100.0, 3.0);
        assert_relative_eq!(tf.num, dvector![250594.32325190006], epsilon = 1e-9);
        assert_relative_eq!(
            tf.den,
            dvector![
                1.0,
                59.72404165413484,
                9283.480575752415,
                250594.32325190003
            ],
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_chebyshev2() {
        let tf = chebyshev2(1, 100.0, 1.0);
        assert_relative_eq!(tf.num, dvector![196.52267283602717]);
        assert_relative_eq!(tf.den, dvector![1.0, 196.52267283602717]);

        let tf = chebyshev2(2, 100.0, 1.0);
        assert_relative_eq!(
            tf.num,
            dvector![0.8912509381337451, 0.0, 17825.018762674903],
            epsilon = 1e-10
        );
        assert_relative_eq!(tf.den, dvector![1.0, 62.26482262384244, 17825.01876267491]);

        let tf = chebyshev2(3, 100.0, 1.0);
        assert_relative_eq!(
            tf.num,
            dvector![589.5680185080812, 0.0, 7860906.913441084],
            epsilon = 1e-8
        );
        assert_relative_eq!(
            tf.den,
            dvector![1.0, 631.729847643468, 25746.0759780469, 7860906.913441085],
            epsilon = 1e-8
        );
    }
}
