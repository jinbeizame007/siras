use nalgebra::{stack, DVector};

pub fn anti_symmetric_reflect_extension(x: DVector<f64>) -> DVector<f64> {
    let x_reversed = DVector::from_iterator(x.len(), x.as_slice().iter().rev().copied());
    let x_padded = stack![
        -&x_reversed.clone().add_scalar(x[0] * 2.0);
        &x;
        -&x_reversed.clone().add_scalar(x[x.len() - 1] * 2.0)
    ];
    x_padded
}
