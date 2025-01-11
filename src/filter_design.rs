use std::f64::consts::PI;

use nalgebra::{dvector, Complex, DVector};

use crate::transfer_function::{convolve, ContinuousTransferFunction};

pub fn butter(order: usize, cutoff_frequency: f64) -> ContinuousTransferFunction {
    let num = dvector![cutoff_frequency.powf(order as f64)];

    let thetas: Vec<f64> = (1..=order)
        .map(|k| PI * (2 * k + order - 1) as f64 / (2 * order) as f64)
        .collect();
    let mut poles: Vec<Complex<f64>> = thetas
        .iter()
        .map(|theta| cutoff_frequency * Complex::new(theta.cos(), theta.sin()))
        .collect();
    poles = poles.iter().filter(|p| p.re <= 0.0).cloned().collect();
    let den_complex = poly(DVector::from_vec(poles));
    let den = DVector::from_vec(den_complex.iter().map(|e| e.re).collect::<Vec<_>>());

    ContinuousTransferFunction::new(num, den)
}

pub fn poly(vec: DVector<Complex<f64>>) -> DVector<Complex<f64>> {
    let mut a = DVector::from_vec(vec![Complex::new(1.0, 0.0)]);
    for x in vec.iter() {
        a = convolve(&a, &DVector::from_vec(vec![Complex::new(1.0, 0.0), -x]));
    }

    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_butter() {
        let tf = butter(2, 1.0);
        assert_eq!(tf.num, dvector![1.0]);
        assert_relative_eq!(tf.den, dvector![1.0, 1.414214, 1.0], epsilon = 1e-5);
    }

    #[test]
    fn test_poly() {
        let vec = dvector![
            Complex::new(1.0, 0.0),
            Complex::new(2.0, 0.0),
            Complex::new(3.0, 0.0)
        ];
        let result = poly(vec);
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
}
