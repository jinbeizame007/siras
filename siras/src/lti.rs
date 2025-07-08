use nalgebra::{stack, DMatrix, DVector};

use crate::filter_design::{
    design_bessel, design_butter, design_chebyshev1, design_chebyshev2, digital_to_analog_cutoff,
    BandType,
};
use crate::math::expm;
use crate::signal_extension::anti_symmetric_reflect_extension;

const DEFAULT_ALPHA: f64 = 0.5;

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

    pub fn butter(order: usize, cutoff_freq: f64, filter_type: BandType) -> Self {
        design_butter(order, cutoff_freq, filter_type)
    }

    pub fn bessel(order: usize, cutoff_freq: f64, filter_type: BandType) -> Self {
        design_bessel(order, cutoff_freq, filter_type)
    }

    pub fn chebyshev1(
        order: usize,
        cutoff_freq: f64,
        ripple_db: f64,
        filter_type: BandType,
    ) -> Self {
        design_chebyshev1(order, cutoff_freq, ripple_db, filter_type)
    }

    pub fn chebyshev2(
        order: usize,
        cutoff_freq: f64,
        ripple_db: f64,
        filter_type: BandType,
    ) -> Self {
        design_chebyshev2(order, cutoff_freq, ripple_db, filter_type)
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
    pub num: DVector<f64>,
    pub den: DVector<f64>,
    inputs: DVector<f64>,
    outputs: DVector<f64>,
    #[allow(unused)]
    pub dt: f64,
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

    pub fn butter(order: usize, cutoff_freq: f64, dt: f64, filter_type: BandType) -> Self {
        let sample_rate = 1.0 / dt;
        let normalized_cutoff_freq = digital_to_analog_cutoff(cutoff_freq, sample_rate);
        design_butter(order, normalized_cutoff_freq, filter_type).to_discrete(dt, DEFAULT_ALPHA)
    }

    pub fn bessel(order: usize, cutoff_freq: f64, dt: f64, filter_type: BandType) -> Self {
        let sample_rate = 1.0 / dt;
        let normalized_cutoff_freq = digital_to_analog_cutoff(cutoff_freq, sample_rate);
        design_bessel(order, normalized_cutoff_freq, filter_type).to_discrete(dt, DEFAULT_ALPHA)
    }

    pub fn chebyshev1(
        order: usize,
        cutoff_freq: f64,
        ripple_db: f64,
        dt: f64,
        filter_type: BandType,
    ) -> Self {
        let sample_rate = 1.0 / dt;
        let normalized_cutoff_freq = digital_to_analog_cutoff(cutoff_freq, sample_rate);
        design_chebyshev1(order, normalized_cutoff_freq, ripple_db, filter_type)
            .to_discrete(dt, DEFAULT_ALPHA)
    }

    pub fn chebyshev2(
        order: usize,
        cutoff_freq: f64,
        ripple_db: f64,
        dt: f64,
        filter_type: BandType,
    ) -> Self {
        let sample_rate = 1.0 / dt;
        let normalized_cutoff_freq = digital_to_analog_cutoff(cutoff_freq, sample_rate);
        design_chebyshev2(order, normalized_cutoff_freq, ripple_db, filter_type)
            .to_discrete(dt, DEFAULT_ALPHA)
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
        let freq1 = 10.0;
        let freq2 = 100.0;
        let sample_rate = 32000;
        let t = DVector::from_iterator(
            sample_rate + 1,
            (0..=sample_rate).map(|i| i as f64 / sample_rate as f64),
        );
        let low_frequency_sin_wave = (2.0 * PI * freq1 * t.clone()).map(|e| e.sin());
        let high_frequency_sin_wave = (2.0 * PI * freq2 * t.clone()).map(|e| e.sin());
        let x = low_frequency_sin_wave.clone() + high_frequency_sin_wave;

        // Chebyshev Type I Filter
        // let order = 4;
        // let cutoff_freq = 90.0;
        // let ripple = 0.1;
        // let mut tf = chebyshev1(order, cutoff_freq, ripple);
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
}
