use nalgebra::{stack, DMatrix, DVector};

use crate::lti::{
    ContinuousStateSpace, ContinuousTransferFunction, DiscreteStateSpace, DiscreteTransferFunction,
};
use crate::math::characteristic_polynomial;

impl From<ContinuousStateSpace> for ContinuousTransferFunction {
    fn from(state_space: ContinuousStateSpace) -> Self {
        let a = state_space.a.clone();
        let b = state_space.b.clone();
        let c = state_space.c.clone();
        let d = state_space.d.clone();

        let o_den = characteristic_polynomial(&a);
        let den = match o_den {
            Some(den) => den,
            None => DVector::from_vec(vec![1.0]),
        };
        let num =
            characteristic_polynomial(&(a - (&b * &c))).unwrap() + den.clone() * d.add_scalar(-1.0);

        Self::new(num, den)
    }
}

impl From<DiscreteStateSpace> for DiscreteTransferFunction {
    fn from(state_space: DiscreteStateSpace) -> Self {
        let a = state_space.a.clone();
        let b = state_space.b.clone();
        let c = state_space.c.clone();
        let d = state_space.d.clone();
        let dt = state_space.dt;

        let o_den = characteristic_polynomial(&a);
        let den = match o_den {
            Some(den) => den,
            None => DVector::from_vec(vec![1.0]),
        };
        let num =
            characteristic_polynomial(&(a - (&b * &c))).unwrap() + den.clone() * d.add_scalar(-1.0);

        Self::new(num, den, dt)
    }
}

impl From<ContinuousTransferFunction> for ContinuousStateSpace {
    fn from(tf: ContinuousTransferFunction) -> Self {
        assert!(
            tf.den.len() >= tf.num.len(),
            "The order of the denominator must be greater than or equal to the order of the numerator."
        );

        let n = tf.den.len() - 1; // Order

        // Normalize the numerator and denominator
        let num = stack![DVector::zeros(tf.den.len() - tf.num.len()); tf.num.clone()] / tf.den[0];
        let den = tf.den.clone() / tf.den[0];

        let a = stack![
            -den.rows(1, n).transpose();
            DMatrix::identity(n - 1, n)
        ];
        let b = DMatrix::identity(n, 1);
        let c = DMatrix::from_row_slice(1, n, num.rows(1, n).as_slice())
            - num[0] * DMatrix::from_row_slice(1, n, den.rows(1, n).as_slice());
        let d = DMatrix::from_row_slice(1, 1, &[num[0]]);

        ContinuousStateSpace::new(a, b, c, d)
    }
}

impl From<DiscreteTransferFunction> for DiscreteStateSpace {
    fn from(tf: DiscreteTransferFunction) -> Self {
        assert!(
            tf.den.len() >= tf.num.len(),
            "The order of the denominator must be greater than or equal to the order of the numerator."
        );

        let n = tf.den.len() - 1; // Order

        // Normalize the numerator and denominator
        let num = stack![DVector::zeros(tf.den.len() - tf.num.len()); tf.num.clone()] / tf.den[0];
        let den = tf.den.clone() / tf.den[0];

        let a = stack![
            -den.rows(1, n).transpose();
            DMatrix::identity(n - 1, n)
        ];
        let b = DMatrix::identity(n, 1);
        let c = DMatrix::from_row_slice(1, n, num.rows(1, n).as_slice())
            - num[0] * DMatrix::from_row_slice(1, n, den.rows(1, n).as_slice());
        let d = DMatrix::from_row_slice(1, 1, &[num[0]]);

        DiscreteStateSpace::new(a, b, c, d, tf.dt)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{dmatrix, dvector};

    #[test]
    fn test_continuous_transfer_function_to_continuous_state_space() {
        let num = dvector![1.0, 3.0, 3.0];
        let den = dvector![1.0, 2.0, 1.0];

        let tf = ContinuousTransferFunction::new(num, den);
        let ss = ContinuousStateSpace::from(tf);

        let expected_a = dmatrix![-2.0, -1.0; 1.0, 0.0];
        let expected_b = dmatrix![1.0; 0.0];
        let expected_c = dmatrix![1.0, 2.0];
        let expected_d = dmatrix![1.0];

        assert_relative_eq!(ss.a, expected_a);
        assert_relative_eq!(ss.b, expected_b);
        assert_relative_eq!(ss.c, expected_c);
        assert_relative_eq!(ss.d, expected_d);
    }

    #[test]
    fn test_continuous_transfer_function_to_continuous_state_space_different_order() {
        let num = dvector![1.0, 2.0, 3.0];
        let den = dvector![1.0, 2.0, 3.0, 4.0];
        let tf = ContinuousTransferFunction::new(num, den);
        let ss = ContinuousStateSpace::from(tf);

        let expected_a = dmatrix![-2.0, -3.0, -4.0; 1.0, 0.0, 0.0; 0.0, 1.0, 0.0];
        let expected_b = dmatrix![1.0; 0.0; 0.0];
        let expected_c = dmatrix![1.0, 2.0, 3.0];
        let expected_d = dmatrix![0.0];

        assert_relative_eq!(ss.a, expected_a);
        assert_relative_eq!(ss.b, expected_b);
        assert_relative_eq!(ss.c, expected_c);
        assert_relative_eq!(ss.d, expected_d);
    }

    #[test]
    fn test_continuous_state_space_to_discrete_state_space() {
        let ac = dmatrix![1.0, 0.0; 0.0, 1.0];
        let bc = dmatrix![0.5; 0.5];
        let cc = dmatrix![0.75, 1.0; 1.0, 1.0; 1.0, 0.25];
        let dc = dmatrix![0.0; 0.0; -0.33];
        let continuous_state_space = ContinuousStateSpace::new(ac, bc, cc, dc);

        let dt = 0.5;
        let alpha = 1.0 / 3.0;
        let discrete_state_space = continuous_state_space.to_discrete(dt, alpha);

        let expected_a = dmatrix![1.6, 0.0; 0.0, 1.6];
        let expected_b = dmatrix![0.3; 0.3];
        let expected_c = dmatrix![0.9, 1.2; 1.2, 1.2; 1.2, 0.3];
        let expected_d = dmatrix![0.175; 0.2; -0.205];

        assert_relative_eq!(discrete_state_space.a, expected_a);
        assert_relative_eq!(discrete_state_space.b, expected_b);
        assert_relative_eq!(discrete_state_space.c, expected_c);
        assert_relative_eq!(discrete_state_space.d, expected_d);
    }

    #[test]
    fn test_continuous_state_space_to_continuous_transfer_function() {
        let num = DVector::from_vec(vec![1.0, 3.0, 5.0]);
        let den = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let continuous_transfer_function = ContinuousTransferFunction::from(
            ContinuousStateSpace::from(ContinuousTransferFunction::new(num.clone(), den.clone())),
        );

        assert_relative_eq!(continuous_transfer_function.num, num);
        assert_relative_eq!(continuous_transfer_function.den, den);
    }

    #[test]
    fn test_discrete_state_space_to_discrete_transfer_function() {
        let a = dmatrix![-2.0, -1.0; 1.0, 0.0];
        let b = dmatrix![1.0; 0.0];
        let c = dmatrix![1.0, 2.0];
        let d = dmatrix![1.0];
        let dt = 0.1;
        let discrete_state_space = DiscreteStateSpace::new(a, b, c, d, dt);
        let discrete_transfer_function = DiscreteTransferFunction::from(discrete_state_space);

        let expected_num = dvector![1.0, 3.0, 3.0];
        let expected_den = dvector![1.0, 2.0, 1.0];

        assert_relative_eq!(
            discrete_transfer_function.num,
            expected_num,
            epsilon = 1e-15
        );
        assert_relative_eq!(
            discrete_transfer_function.den,
            expected_den,
            epsilon = 1e-15
        );
    }
}
