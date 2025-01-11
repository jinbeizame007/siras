use std::f64::consts::PI;

use nalgebra::{dvector, Complex, DVector};

use crate::transfer_function::{convolve, ContinuousTransferFunction};

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
