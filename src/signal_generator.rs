use std::f64::consts::PI;

use nalgebra::DVector;

pub fn generate_sine_wave(t: &DVector<f64>, freq: f64) -> DVector<f64> {
    (2.0 * PI * freq * t.clone()).map(|e| e.sin())
}
