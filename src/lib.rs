use nalgebra::{stack, DMatrix, DVector};

pub struct ContinuousTransferFunction {
    num: DVector<f64>,
    den: DVector<f64>,
    dt: f64,
}

impl ContinuousTransferFunction {
    pub fn new(num: DVector<f64>, den: DVector<f64>, dt: f64) -> Self {
        Self { num, den, dt }
    }
}

pub struct DiscreteTransferFunction {
    num: DVector<f64>,
    den: DVector<f64>,
    inputs: DVector<f64>,
    outputs: DVector<f64>,
    dt: f64,
}

impl DiscreteTransferFunction {
    pub fn new(num: DVector<f64>, den: DVector<f64>, dt: f64) -> Self {
        let inputs = DVector::zeros(num.len());
        let outputs = DVector::zeros(den.len());

        Self {
            num,
            den,
            inputs,
            outputs,
            dt,
        }
    }

    pub fn step(&mut self, input: f64) -> f64 {
        let mut output = 0.0;

        for i in (1..self.inputs.len()).rev() {
            self.inputs[i] = self.inputs[i - 1];
        }
        self.inputs[0] = input;
        output += self.num.dot(&self.inputs);

        for i in (1..self.outputs.len()).rev() {
            self.outputs[i] = self.outputs[i - 1];
        }
        output -= self
            .den
            .rows(1, self.den.len() - 1)
            .dot(&self.outputs.rows(1, self.outputs.len() - 1));
        output /= self.den[0];
        self.outputs[0] = output;

        output
    }
}

pub struct ContinuousStateSpace {
    pub a: DMatrix<f64>,
    pub b: DMatrix<f64>,
    pub c: DMatrix<f64>,
    pub d: DMatrix<f64>,
}

impl ContinuousStateSpace {
    pub fn new(a: DMatrix<f64>, b: DMatrix<f64>, c: DMatrix<f64>, d: DMatrix<f64>) -> Self {
        Self { a, b, c, d }
    }

    pub fn to_discrete(&self, dt: f64, alpha: f64) -> DiscreteStateSpace {
        let a = self.a.clone();
        let b = self.b.clone();
        let c = self.c.clone();
        let d = self.d.clone();

        let ima = DMatrix::identity(a.nrows(), a.nrows()) - alpha * dt * &a;
        let ima_lu = ima.clone().lu();
        let ad = ima_lu
            .solve(&(DMatrix::identity(a.nrows(), a.nrows()) + (1.0 - alpha) * dt * &a))
            .unwrap();
        let bd = ima_lu.solve(&(dt * &b)).unwrap();
        let cd = ima
            .transpose()
            .lu()
            .solve(&c.transpose())
            .unwrap()
            .transpose();
        let dd = d + alpha * (&c * &bd);

        DiscreteStateSpace {
            a: ad,
            b: bd,
            c: cd,
            d: dd,
            dt,
        }
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
            - num[0] * DMatrix::from_row_slice(1, n, &den.rows(1, n).as_slice());
        let d = DMatrix::from_row_slice(1, 1, &[num[0]]);
        ContinuousStateSpace { a, b, c, d }
    }
}

pub struct DiscreteStateSpace {
    pub a: DMatrix<f64>,
    pub b: DMatrix<f64>,
    pub c: DMatrix<f64>,
    pub d: DMatrix<f64>,
    pub dt: f64,
}

impl DiscreteStateSpace {
    pub fn new(
        a: DMatrix<f64>,
        b: DMatrix<f64>,
        c: DMatrix<f64>,
        d: DMatrix<f64>,
        dt: f64,
    ) -> Self {
        Self { a, b, c, d, dt }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_continuous_transfer_function_to_continuous_state_space() {
        let num = DVector::from_vec(vec![1.0, 3.0, 3.0]);
        let den = DVector::from_vec(vec![1.0, 2.0, 1.0]);
        let dt = 0.1;

        let tf = ContinuousTransferFunction::new(num, den, dt);
        let ss = ContinuousStateSpace::from(tf);

        let expected_a = DMatrix::from_row_slice(2, 2, &[-2.0, -1.0, 1.0, 0.0]);
        let expected_b = DMatrix::from_row_slice(2, 1, &[1.0, 0.0]);
        let expected_c = DMatrix::from_row_slice(1, 2, &[1.0, 2.0]);
        let expected_d = DMatrix::from_row_slice(1, 1, &[1.0]);

        assert_relative_eq!(ss.a, expected_a);
        assert_relative_eq!(ss.b, expected_b);
        assert_relative_eq!(ss.c, expected_c);
        assert_relative_eq!(ss.d, expected_d);
    }

    #[test]
    fn test_continuous_transfer_function_to_continuous_state_space_different_order() {
        let num = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let den = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let dt = 0.1;
        let tf = ContinuousTransferFunction::new(num, den, dt);
        let ss = ContinuousStateSpace::from(tf);

        let expected_a =
            DMatrix::from_row_slice(3, 3, &[-2.0, -3.0, -4.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let expected_b = DMatrix::from_row_slice(3, 1, &[1.0, 0.0, 0.0]);
        let expected_c = DMatrix::from_row_slice(1, 3, &[1.0, 2.0, 3.0]);
        let expected_d = DMatrix::from_row_slice(1, 1, &[0.0]);

        assert_relative_eq!(ss.a, expected_a);
        assert_relative_eq!(ss.b, expected_b);
        assert_relative_eq!(ss.c, expected_c);
        assert_relative_eq!(ss.d, expected_d);
    }

    #[test]
    fn test_continuous_state_space_to_discrete_state_space() {
        let ac = DMatrix::identity(2, 2);
        let bc = DMatrix::from_row_slice(2, 1, &[0.5, 0.5]);
        let cc = DMatrix::from_row_slice(3, 2, &[0.75, 1.0, 1.0, 1.0, 1.0, 0.25]);
        let dc = DMatrix::from_row_slice(3, 1, &[0.0, 0.0, -0.33]);
        let continuous_state_space = ContinuousStateSpace::new(ac, bc, cc, dc);

        let dt = 0.5;
        let alpha = 1.0 / 3.0;
        let discrete_state_space = continuous_state_space.to_discrete(dt, alpha);

        let expected_a = 1.6 * DMatrix::identity(2, 2);
        let expected_b = DMatrix::from_row_slice(2, 1, &[0.3, 0.3]);
        let expected_c = DMatrix::from_row_slice(3, 2, &[0.9, 1.2, 1.2, 1.2, 1.2, 0.3]);
        let expected_d = DMatrix::from_row_slice(3, 1, &[0.175, 0.2, -0.205]);

        assert_relative_eq!(discrete_state_space.a, expected_a);
        assert_relative_eq!(discrete_state_space.b, expected_b);
        assert_relative_eq!(discrete_state_space.c, expected_c);
        assert_relative_eq!(discrete_state_space.d, expected_d);
    }
}
