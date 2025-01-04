use nalgebra::DMatrix;

pub struct ContinuousTransferFunction {
    num: Vec<f64>,
    den: Vec<f64>,
}

impl ContinuousTransferFunction {
    pub fn new(num: Vec<f64>, den: Vec<f64>) -> Self {
        Self { num, den }
    }
}

pub struct DiscreteTransferFunction {
    num: Vec<f64>,
    den: Vec<f64>,
    inputs: Vec<f64>,
    outputs: Vec<f64>,
}

impl DiscreteTransferFunction {
    pub fn new(num: Vec<f64>, den: Vec<f64>) -> Self {
        let inputs = vec![0.0; num.len()];
        let outputs = vec![0.0; den.len()];
        Self { num, den, inputs, outputs }
    }

    pub fn step(&mut self, input: f64) -> f64 {
        let mut output = 0.0;

        self.num.rotate_right(1);
        self.num[0] = input;
        for i in 0..self.num.len() {
            output += self.num[i] * self.inputs[i];
        }

        self.den.rotate_right(1);
        for i in 1..self.den.len() {
            output -= self.den[i] * self.outputs[i];
        }
        output /= self.den[0];
        self.outputs[0] = output;

        output
    }
}

pub struct StateSpace {
    a: DMatrix<f64>,
    b: DMatrix<f64>,
    c: DMatrix<f64>,
    d: DMatrix<f64>,
}

impl StateSpace {
    pub fn new(a: DMatrix<f64>, b: DMatrix<f64>, c: DMatrix<f64>, d: DMatrix<f64>) -> Self {
        Self { a, b, c, d }
    }
}
