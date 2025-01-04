use nalgebra::{stack, DMatrix, DVector};

pub struct ContinuousTransferFunction {
    num: DVector<f64>,
    den: DVector<f64>,
}

impl ContinuousTransferFunction {
    pub fn new(num: DVector<f64>, den: DVector<f64>) -> Self {
        Self { num, den }
    }
}

pub struct DiscreteTransferFunction {
    num: DVector<f64>,
    den: DVector<f64>,
    inputs: DVector<f64>,
    outputs: DVector<f64>,
}

impl DiscreteTransferFunction {
    pub fn new(num: DVector<f64>, den: DVector<f64>) -> Self {
        let inputs = DVector::zeros(num.len());
        let outputs = DVector::zeros(den.len());

        Self {
            num,
            den,
            inputs,
            outputs,
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

pub struct StateSpace {
    #[allow(dead_code)]
    a: DMatrix<f64>,
    #[allow(dead_code)]
    b: DMatrix<f64>,
    #[allow(dead_code)]
    c: DMatrix<f64>,
    #[allow(dead_code)]
    d: DMatrix<f64>,
}

impl StateSpace {
    pub fn new(a: DMatrix<f64>, b: DMatrix<f64>, c: DMatrix<f64>, d: DMatrix<f64>) -> Self {
        Self { a, b, c, d }
    }
}

impl From<ContinuousTransferFunction> for StateSpace {
    fn from(tf: ContinuousTransferFunction) -> Self {
        // Normalize the numerator and denominator
        let num = tf.num.clone() / tf.num[0];
        let den = tf.den.clone() / tf.den[0];

        let a = stack![
            -den.rows(1, den.len() - 1),
            DMatrix::identity(den.len() - 1, den.len() - 1)
        ];
        let b = DMatrix::identity(den.len() - 1, 1);
        let c = DMatrix::from_vec(
            1,
            num.len() - 1,
            num.rows(0, num.len() - 2).as_slice().to_vec(),
        );
        let d = DMatrix::zeros(c.nrows(), b.ncols());

        StateSpace { a, b, c, d }
    }
}
