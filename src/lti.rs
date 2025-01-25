use nalgebra::{stack, DMatrix, DVector};

use crate::filter_design::{
    design_bessel, design_butter, design_chebyshev1, design_chebyshev2, digital_to_analog_cutoff,
    FilterType,
};
use crate::math::{characteristic_polynomial, expm};
use crate::signal_extension::anti_symmetric_reflect_extension;

pub trait LTI {
    fn reset(&mut self);
    fn filtfilt(&mut self, u: &DVector<f64>, t: &DVector<f64>) -> DVector<f64>;
    fn simulate(&mut self, inputs: DVector<f64>, t: DVector<f64>) -> DVector<f64>;
}

#[derive(Clone, Debug)]
pub struct ContinuousTransferFunction {
    pub num: DVector<f64>,
    pub den: DVector<f64>,
    x: DVector<f64>,
}

impl ContinuousTransferFunction {
    pub fn new(num: DVector<f64>, den: DVector<f64>) -> Self {
        let x = DVector::zeros(num.len());
        Self { num, den, x }
    }

    pub fn butter(order: usize, cutoff_frequency: f64, filter_type: FilterType) -> Self {
        design_butter(order, cutoff_frequency, filter_type)
    }

    pub fn bessel(order: usize, cutoff_frequency: f64, filter_type: FilterType) -> Self {
        design_bessel(order, cutoff_frequency, filter_type)
    }

    pub fn chebyshev1(
        order: usize,
        cutoff_frequency: f64,
        ripple_db: f64,
        filter_type: FilterType,
    ) -> Self {
        design_chebyshev1(order, cutoff_frequency, ripple_db, filter_type)
    }

    pub fn chebyshev2(
        order: usize,
        cutoff_frequency: f64,
        ripple_db: f64,
        filter_type: FilterType,
    ) -> Self {
        design_chebyshev2(order, cutoff_frequency, ripple_db, filter_type)
    }

    pub fn reset(&mut self) {
        LTI::reset(self)
    }

    pub fn simulate(&mut self, inputs: DVector<f64>, t: DVector<f64>) -> DVector<f64> {
        LTI::simulate(self, inputs, t)
    }

    pub fn filtfilt(&mut self, u: &DVector<f64>, t: &DVector<f64>) -> DVector<f64> {
        LTI::filtfilt(self, u, t)
    }

    pub fn impulse(&self, t: DVector<f64>) -> DVector<f64> {
        let state_space = ContinuousStateSpace::from(self.clone());

        state_space.impulse(t)
    }

    pub fn to_discrete(&self, dt: f64, alpha: f64) -> DiscreteTransferFunction {
        let state_space = ContinuousStateSpace::from(self.clone());
        let discrete_state_space = state_space.to_discrete(dt, alpha);
        DiscreteTransferFunction::from(discrete_state_space)
    }
}

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

impl LTI for ContinuousTransferFunction {
    fn reset(&mut self) {
        self.x = DVector::zeros(self.num.len());
    }

    fn filtfilt(&mut self, u: &DVector<f64>, t: &DVector<f64>) -> DVector<f64> {
        let mut state_space = ContinuousStateSpace::from(self.clone());

        let result = state_space.filtfilt(u, t);
        self.x = state_space.x.clone();

        result
    }

    fn simulate(&mut self, inputs: DVector<f64>, t: DVector<f64>) -> DVector<f64> {
        let mut state_space = ContinuousStateSpace::from(self.clone());

        let result = state_space.simulate(inputs, t);
        self.x = state_space.x.clone();

        result
    }
}

#[derive(Clone, Debug)]
pub struct DiscreteTransferFunction {
    num: DVector<f64>,
    den: DVector<f64>,
    inputs: DVector<f64>,
    outputs: DVector<f64>,
    #[allow(unused)]
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

    pub fn butter(
        order: usize,
        cutoff_frequency: f64,
        dt: f64,
        alpha: f64,
        filter_type: FilterType,
    ) -> Self {
        let sample_frequency = 1.0 / dt;
        let normalized_cutoff_frequency =
            digital_to_analog_cutoff(cutoff_frequency, sample_frequency);
        design_butter(order, normalized_cutoff_frequency, filter_type).to_discrete(dt, alpha)
    }

    pub fn bessel(
        order: usize,
        cutoff_frequency: f64,
        dt: f64,
        alpha: f64,
        filter_type: FilterType,
    ) -> Self {
        let sample_frequency = 1.0 / dt;
        let normalized_cutoff_frequency =
            digital_to_analog_cutoff(cutoff_frequency, sample_frequency);
        design_bessel(order, normalized_cutoff_frequency, filter_type).to_discrete(dt, alpha)
    }

    pub fn chebyshev1(
        order: usize,
        cutoff_frequency: f64,
        ripple_db: f64,
        dt: f64,
        alpha: f64,
        filter_type: FilterType,
    ) -> Self {
        let sample_frequency = 1.0 / dt;
        let normalized_cutoff_frequency =
            digital_to_analog_cutoff(cutoff_frequency, sample_frequency);
        design_chebyshev1(order, normalized_cutoff_frequency, ripple_db, filter_type)
            .to_discrete(dt, alpha)
    }

    pub fn chebyshev2(
        order: usize,
        cutoff_frequency: f64,
        ripple_db: f64,
        dt: f64,
        alpha: f64,
        filter_type: FilterType,
    ) -> Self {
        let sample_frequency = 1.0 / dt;
        let normalized_cutoff_frequency =
            digital_to_analog_cutoff(cutoff_frequency, sample_frequency);
        design_chebyshev2(order, normalized_cutoff_frequency, ripple_db, filter_type)
            .to_discrete(dt, alpha)
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

    pub fn simulate(&mut self, inputs: DVector<f64>, t: DVector<f64>) -> DVector<f64> {
        LTI::simulate(self, inputs, t)
    }

    pub fn filtfilt(&mut self, u: &DVector<f64>, t: &DVector<f64>) -> DVector<f64> {
        LTI::filtfilt(self, u, t)
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

impl LTI for DiscreteTransferFunction {
    fn reset(&mut self) {
        self.inputs = DVector::zeros(self.inputs.len());
        self.outputs = DVector::zeros(self.outputs.len());
    }

    fn simulate(&mut self, inputs: DVector<f64>, t: DVector<f64>) -> DVector<f64> {
        let mut output = DVector::zeros(inputs.len());

        for i in 0..inputs.len() {
            output[i] = self.step(inputs[i]);
        }

        output
    }

    fn filtfilt(&mut self, u: &DVector<f64>, t: &DVector<f64>) -> DVector<f64> {
        // padding
        let u_extended = anti_symmetric_reflect_extension(u.clone());
        let dt = t[1] - t[0];
        let t_extended = stack![t; DVector::from_iterator(t.len() * 2, (1..=t.len() * 2).map(|i| t[0] + i as f64 * dt))];

        // forward filtering
        let y_extended = self.simulate(u_extended.clone(), t_extended.clone());

        // backward filtering
        let mut y_extended = DVector::from_iterator(
            y_extended.len(),
            y_extended.as_slice().iter().rev().copied(),
        );
        y_extended = self.simulate(y_extended, t_extended.clone());
        y_extended = DVector::from_iterator(
            y_extended.len(),
            y_extended.as_slice().iter().rev().copied(),
        );
        let y = y_extended.rows(u.nrows(), u.nrows()).into_owned();

        y
    }
}

#[derive(Clone, Debug)]
pub struct ContinuousStateSpace {
    pub a: DMatrix<f64>,
    pub b: DMatrix<f64>,
    pub c: DMatrix<f64>,
    pub d: DMatrix<f64>,
    pub x: DVector<f64>,
}

impl ContinuousStateSpace {
    pub fn new(a: DMatrix<f64>, b: DMatrix<f64>, c: DMatrix<f64>, d: DMatrix<f64>) -> Self {
        let x = DVector::zeros(a.nrows());
        Self { a, b, c, d, x }
    }

    pub fn filtfilt(&mut self, u: &DVector<f64>, t: &DVector<f64>) -> DVector<f64> {
        // padding
        let u_extended = anti_symmetric_reflect_extension(u.clone());
        let dt = t[1] - t[0];
        let t_extended = stack![t; DVector::from_iterator(t.len() * 2, (1..=t.len() * 2).map(|i| t[0] + i as f64 * dt))];

        // forward filtering
        let y_extended = self.simulate(u_extended.clone(), t_extended.clone());

        // backward filtering
        let mut y_extended = DVector::from_iterator(
            y_extended.len(),
            y_extended.as_slice().iter().rev().copied(),
        );
        y_extended = self.simulate(y_extended, t_extended.clone());
        y_extended = DVector::from_iterator(
            y_extended.len(),
            y_extended.as_slice().iter().rev().copied(),
        );
        let y = y_extended.rows(u.nrows(), u.nrows()).into_owned();

        y
    }

    pub fn simulate(&mut self, inputs: DVector<f64>, t: DVector<f64>) -> DVector<f64> {
        let n_states = self.a.nrows();
        let n_inputs = self.b.ncols();

        let mut xout = DMatrix::<f64>::zeros(t.len(), n_states);
        xout.set_row(0, &self.x.transpose());

        let dt = t[1] - t[0];

        let m = stack![
            stack![self.a.clone() * dt, self.b.clone() * dt, DMatrix::zeros(n_states, n_inputs)];
            stack![DMatrix::zeros(n_inputs, n_states + n_inputs), DMatrix::identity(n_inputs, n_inputs)];
            DMatrix::zeros(n_inputs, n_states + 2 * n_inputs);
        ];

        let exp_mt = expm(&m.transpose());
        let ad = exp_mt.view((0, 0), (n_states, n_states));
        let bd1 = exp_mt.view(
            (n_states + n_inputs, 0),
            (m.nrows() - n_states - n_inputs, n_states),
        );
        let bd0 = exp_mt.view((n_states, 0), (n_inputs, n_states)) - bd1;

        for i in 1..t.len() {
            xout.set_row(
                i,
                &(xout.row(i - 1) * ad + inputs[i - 1] * &bd0 + inputs[i] * bd1),
            );
        }

        DVector::from_column_slice((xout.clone() * self.c.transpose()).column(0).as_slice())
    }

    pub fn impulse(&self, t: DVector<f64>) -> DVector<f64> {
        let mut state_space = self.clone();
        state_space.x = self.b.column(0).into();

        let inputs = DVector::from_element(t.len(), 0.0);

        state_space.simulate(inputs, t)
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

        DiscreteStateSpace::new(ad, bd, cd, dd, dt)
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

#[derive(Clone, Debug)]
pub struct DiscreteStateSpace {
    pub a: DMatrix<f64>,
    pub b: DMatrix<f64>,
    pub c: DMatrix<f64>,
    pub d: DMatrix<f64>,
    pub x: DVector<f64>,
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
        let x = DVector::zeros(a.nrows());
        Self { a, b, c, d, x, dt }
    }

    pub fn step(&mut self, input: f64) -> f64 {
        let output = &self.c * &self.x + &self.d * input;
        self.x = &self.a * &self.x + &self.b * input;

        output[0]
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
    use std::f64::consts::PI;

    #[test]
    fn test_step_discrete_transfer_function() {
        let num = dvector![1.3];
        let den = dvector![2.0, 1.5];
        let dt = 0.1;
        let mut discrete_tf = DiscreteTransferFunction::new(num, den, dt);

        let inputs = [0.2, 0.4, 0.6, 0.8, 1.0];
        let outputs = inputs
            .iter()
            .map(|input| discrete_tf.step(*input))
            .collect::<Vec<_>>();
        let expected_outputs = [0.13, 0.1625, 0.268125, 0.31890625, 0.4108203125];

        for (output, expected_output) in outputs.iter().zip(expected_outputs.iter()) {
            assert_relative_eq!(output, expected_output);
        }
    }

    #[test]
    fn test_step_continuous_state_space() {
        let a = dmatrix![-1.0, 0.0; 0.0, -2.0];
        let b = dmatrix![1.0; 0.0];
        let c = dmatrix![1.0, 0.0];
        let d = dmatrix![1.0];
        let x = dvector![1.0, 1.0];
        let mut continuous_state_space = ContinuousStateSpace { a, b, c, d, x };

        let inputs = DVector::zeros(5);
        let t = dvector![0.0, 0.5, 1.0, 1.5, 2.0];
        let outputs = continuous_state_space.simulate(inputs, t);

        assert_relative_eq!(outputs[4], f64::exp(-2.0));
    }

    #[test]
    fn test_impulse_continuous_state_space() {
        let a = dmatrix![-1.0, 0.0; 0.0, -2.0];
        let b = dmatrix![0.5; 0.5];
        let c = dmatrix![0.75, 1.0];
        let d = dmatrix![-0.33];
        let t = dvector![0.0, 0.25, 0.5, 0.75, 1.0];
        let continuous_state_space = ContinuousStateSpace::new(a, b, c, d);
        let response = continuous_state_space.impulse(t);

        let expected_response = dvector![
            0.875,
            0.5953156235080935,
            0.41138871797795873,
            0.2887025373520954,
            0.20562243205759723
        ];

        assert_relative_eq!(response, expected_response);
    }

    #[test]
    fn test_step_discrete_state_space() {
        let a = dmatrix![-2.0, -3.0; 1.0, 0.0];
        let b = dmatrix![1.0; 0.0];
        let c = dmatrix![1.0, 2.0];
        let d = dmatrix![2.0];
        let dt = 0.1;
        let mut discrete_state_space = DiscreteStateSpace::new(a, b, c, d, dt);

        let inputs = [0.2, 0.4, 0.6, 0.8, 1.0];
        let outputs = inputs
            .iter()
            .map(|input| discrete_state_space.step(*input))
            .collect::<Vec<_>>();
        let expected_outputs = [0.4, 1.0, 1.6, 1.6, 2.8];

        for (output, expected_output) in outputs.iter().zip(expected_outputs.iter()) {
            assert_relative_eq!(output, expected_output);
        }
    }

    #[test]
    fn test_filtfilt_continuous_transfer_function() {
        let f0 = 10.0;
        let f1 = 100.0;
        let sample_frequency = 32000;
        let t = DVector::from_iterator(
            sample_frequency + 1,
            (0..=sample_frequency).map(|i| i as f64 / sample_frequency as f64),
        );
        let low_frequency_sin_wave = (2.0 * PI * f0 * t.clone()).map(|e| e.sin());
        let high_frequency_sin_wave = (2.0 * PI * f1 * t.clone()).map(|e| e.sin());
        let x = low_frequency_sin_wave.clone() + high_frequency_sin_wave;

        // Chebyshev Type I Filter
        // let order = 4;
        // let cutoff_frequency = 90.0;
        // let ripple = 0.1;
        // let mut tf = chebyshev1(order, cutoff_frequency, ripple);
        let num = dvector![53736256.63180374];
        let den = dvector![
            1.0,
            162.33952536509088,
            21277.06074788149,
            1476589.879470203,
            54358493.15756986,
        ];
        let mut tf = ContinuousTransferFunction::new(num, den);
        let y = tf.filtfilt(&x, &t);

        assert_relative_eq!(y, low_frequency_sin_wave, epsilon = 0.03);
    }

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
