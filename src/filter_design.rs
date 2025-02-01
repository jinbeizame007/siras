use std::f64::consts::PI;

use nalgebra::{dvector, stack, Complex, DVector};

use crate::lti::ContinuousTransferFunction;
use crate::math::{factorial, polynomial};

pub enum BandType {
    LowPass,
    HighPass,
}

pub fn design_butter(
    order: usize,
    cutoff_freq: f64,
    filter_type: BandType,
) -> ContinuousTransferFunction {
    let num = match filter_type {
        BandType::LowPass => dvector![cutoff_freq.powf(order as f64)],
        BandType::HighPass => stack![dvector![1.0]; DVector::zeros(order)],
    };

    let thetas: Vec<f64> = (1..=order)
        .map(|k| PI * (2 * k + order - 1) as f64 / (2 * order) as f64)
        .collect();

    let poles: Vec<Complex<f64>> = thetas
        .iter()
        .map(|theta| cutoff_freq * Complex::new(theta.cos(), theta.sin()))
        .filter(|p| p.re <= 0.0)
        .collect();

    let den_complex = polynomial(DVector::from_vec(poles));
    let den = DVector::from_vec(den_complex.iter().map(|e| e.re).collect::<Vec<_>>());

    ContinuousTransferFunction::new(num, den)
}

pub fn design_bessel(
    order: usize,
    cutoff_freq: f64,
    filter_type: BandType,
) -> ContinuousTransferFunction {
    let den_bessel = reverse_bessel_polynomial(order);
    let den_cutoff = match filter_type {
        BandType::LowPass => DVector::from_vec(
            (0..=order)
                .rev()
                .map(|k| (1.0 / cutoff_freq).powf(k as f64))
                .collect::<Vec<_>>(),
        ),
        BandType::HighPass => DVector::from_vec(
            (0..=order)
                .rev()
                .map(|k| (cutoff_freq).powf(k as f64))
                .collect::<Vec<_>>(),
        ),
    };

    let den = match filter_type {
        BandType::LowPass => den_bessel.component_mul(&den_cutoff),
        BandType::HighPass => DVector::from_vec(
            den_bessel
                .component_mul(&den_cutoff)
                .iter()
                .rev()
                .cloned()
                .collect::<Vec<f64>>(),
        ),
    };
    let num = match filter_type {
        BandType::LowPass => dvector![den_bessel[den_bessel.len() - 1]],
        BandType::HighPass => {
            stack![dvector![den_bessel[den_bessel.len() - 1]]; DVector::zeros(order)]
        }
    };

    ContinuousTransferFunction::new(num, den)
}

fn reverse_bessel_polynomial(order: usize) -> DVector<f64> {
    let mut coeffs = DVector::zeros(order + 1);
    coeffs[0] = 1.0;
    for k in 0..order {
        coeffs[order - k] = (factorial(2 * order - k)
            / (usize::pow(2, (order - k) as u32) * factorial(k) * factorial(order - k)))
            as f64;
    }

    coeffs
}

pub fn design_chebyshev1(
    order: usize,
    cutoff_freq: f64,
    ripple_db: f64,
    filter_type: BandType,
) -> ContinuousTransferFunction {
    let ripple = f64::sqrt(10.0_f64.powf(ripple_db / 10.0) - 1.0);
    let mut num = match filter_type {
        BandType::LowPass => {
            dvector![cutoff_freq.powf(order as f64) / (2.0_f64.powf(order as f64 - 1.0) * ripple)]
        }
        BandType::HighPass => {
            stack![dvector![1.0 / (2.0_f64.powf(order as f64 - 1.0) * ripple)]; DVector::zeros(order)]
        }
    };

    let mut poles: DVector<Complex<f64>> = DVector::zeros(order);
    match filter_type {
        BandType::LowPass => {
            for k in 1..=order {
                let theta = (PI / 2.0) * (2.0 * k as f64 - 1.0) / order as f64;
                poles[k - 1] = cutoff_freq
                    * Complex::new(
                        -1.0 * (((1.0 / order as f64) * (1.0 / ripple).asinh()).sinh()
                            * theta.sin())
                        .abs(),
                        ((1.0 / order as f64) * (1.0 / ripple).asinh()).cosh() * theta.cos(),
                    )
            }
        }
        BandType::HighPass => {
            for k in 1..=order {
                let theta = (PI / 2.0) * (2.0 * k as f64 - 1.0) / order as f64;
                poles[k - 1] = Complex::new(
                    -1.0 * (((1.0 / order as f64) * (1.0 / ripple).asinh()).sinh() * theta.sin())
                        .abs(),
                    ((1.0 / order as f64) * (1.0 / ripple).asinh()).cosh() * theta.cos(),
                )
            }
            num[0] /= (-poles.clone()).product().re;

            poles = poles.map(|e| e / cutoff_freq);
        }
    }
    let den = match filter_type {
        BandType::LowPass => {
            DVector::from_vec(polynomial(poles).iter().map(|e| e.re).collect::<Vec<_>>())
        }
        BandType::HighPass => {
            let den = DVector::from_vec(
                polynomial(poles)
                    .iter()
                    .rev()
                    .map(|e| e.re)
                    .collect::<Vec<_>>(),
            );
            let den_0 = den[0];

            den.map(|e| e / den_0)
        }
    };

    ContinuousTransferFunction::new(num, den)
}

pub fn chebyshev1_polynomial(order: usize) -> DVector<f64> {
    if order == 0 {
        dvector![1.0]
    } else if order == 1 {
        dvector![1.0, 0.0]
    } else {
        let mut polynomials: Vec<DVector<f64>> = vec![dvector![1.0], dvector![1.0, 0.0]];
        for k in 2..=order {
            polynomials.push(
                stack![2.0 * polynomials[k - 1].clone(); dvector![0.0]]
                    - stack![dvector![0.0, 0.0]; polynomials[k - 2].clone()],
            );
        }

        polynomials[order].clone()
    }
}

pub fn design_chebyshev2(
    order: usize,
    cutoff_freq: f64,
    ripple_db: f64,
    filter_type: BandType,
) -> ContinuousTransferFunction {
    let ripple = 1.0 / f64::sqrt(10.0_f64.powf(ripple_db / 10.0) - 1.0);

    let mut poles_num_vec: Vec<Complex<f64>> = vec![];
    let mut poles_den: DVector<Complex<f64>> = DVector::zeros(order);

    for k in 1..=order {
        let theta = (PI / 2.0) * (2.0 * k as f64 - 1.0) / order as f64;
        if (2 * k - 1) != order {
            poles_num_vec.push(Complex::new(0.0, -1.0 / theta.cos()));
        }
        poles_den[k - 1] = 1.0
            / Complex::new(
                -1.0 * (((1.0 / order as f64) * (1.0 / ripple).asinh()).sinh() * theta.sin()).abs(),
                1.0 * ((1.0 / order as f64) * (1.0 / ripple).asinh()).cosh() * theta.cos(),
            );
    }

    let mut poles_num = DVector::from_vec(poles_num_vec);

    match filter_type {
        BandType::LowPass => {
            poles_den = poles_den.map(|e| e * cutoff_freq);
            poles_num = poles_num.map(|e| e * cutoff_freq);
        }
        BandType::HighPass => {
            poles_den = poles_den.map(|e| cutoff_freq / e);
            poles_num = poles_num.map(|e| cutoff_freq / e);
        }
    }

    let mut num = DVector::from_vec(
        polynomial(poles_num)
            .iter()
            .map(|e| e.re)
            .collect::<Vec<_>>(),
    );
    let den = DVector::from_vec(
        polynomial(poles_den)
            .iter()
            .map(|e| e.re)
            .collect::<Vec<_>>(),
    );

    // Normalize the numerator
    match filter_type {
        BandType::LowPass => {
            num *= den[den.len() - 1] / num[num.len() - 1];
        }
        BandType::HighPass => {
            num /= num[0];
            if order % 2 == 1 {
                num = stack![num; dvector![0.0]];
            }
        }
    }

    ContinuousTransferFunction::new(num, den)
}

pub fn digital_to_analog_cutoff(digital_cutoff: f64, sample_rate: f64) -> f64 {
    2.0 * sample_rate * (PI * digital_cutoff / sample_rate).tan()
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use rstest::rstest;

    #[rstest]
    #[case(1, 1.0, vec![1.0], vec![1.0, 1.0], 1e-15)]
    #[case(2, 1.0, vec![1.0], vec![1.0, f64::sqrt(2.0), 1.0], 1e-15)]
    #[case(3, 1.0, vec![1.0], vec![1.0, 2.0, 2.0, 1.0], 1e-15)]
    #[case(4, 1.0, vec![1.0], vec![1.0, (2.0 + f64::sqrt(2.0)).sqrt() + (2.0 - f64::sqrt(2.0)).sqrt(), 2.0 + (2.0 + f64::sqrt(2.0)).sqrt() * (2.0 - f64::sqrt(2.0)).sqrt(), (2.0 + f64::sqrt(2.0)).sqrt() + (2.0 - f64::sqrt(2.0)).sqrt(), 1.0], 1e-14)]
    fn test_butterworth_low_pass(
        #[case] order: usize,
        #[case] cutoff_freq: f64,
        #[case] expected_num: Vec<f64>,
        #[case] expected_den: Vec<f64>,
        #[case] epsilon: f64,
    ) {
        let tf = design_butter(order, cutoff_freq, BandType::LowPass);
        assert_relative_eq!(tf.num, DVector::from_vec(expected_num), epsilon = epsilon);
        assert_relative_eq!(tf.den, DVector::from_vec(expected_den), epsilon = epsilon);
    }

    #[rstest]
    #[case(1, 1.0, vec![1.0, 0.0], vec![1.0, 1.0], 1e-15)]
    #[case(2, 1.0, vec![1.0, 0.0, 0.0], vec![1.0, f64::sqrt(2.0), 1.0], 1e-15)]
    #[case(2, 10.0, vec![1.0, 0.0, 0.0], vec![1.0, 14.142135623730951, 100.0], 1e-15)]
    #[case(3, 1.0, vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 2.0, 2.0, 1.0], 1e-15)]
    #[case(3, 10.0, vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 20.0, 200.0, 1000.0], 1e-15)]
    #[case(4, 10.0, vec![1.0, 0.0, 0.0, 0.0, 0.0], vec![1.0, 26.131259297527535, 341.4213562373095, 2613.125929752753, 10000.0], 1e-14)]
    fn test_butterworth_high_pass(
        #[case] order: usize,
        #[case] cutoff_freq: f64,
        #[case] expected_num: Vec<f64>,
        #[case] expected_den: Vec<f64>,
        #[case] epsilon: f64,
    ) {
        let tf = design_butter(order, cutoff_freq, BandType::HighPass);
        assert_relative_eq!(tf.num, DVector::from_vec(expected_num), epsilon = epsilon);
        assert_relative_eq!(tf.den, DVector::from_vec(expected_den), epsilon = epsilon);
    }

    #[rstest]
    #[case(1, 1.0, vec![1.0], vec![1.0, 1.0], 1e-15)]
    #[case(2, 1.0, vec![3.0], vec![1.0, 3.0, 3.0], 1e-15)]
    #[case(3, 1.0, vec![15.0], vec![1.0, 6.0, 15.0, 15.0], 1e-15)]
    #[case(4, 1.0, vec![105.0], vec![1.0, 10.0, 45.0, 105.0, 105.0], 1e-15)]
    #[case(5, 1.0, vec![945.0], vec![1.0, 15.0, 105.0, 420.0, 945.0, 945.0], 1e-15)]
    fn test_bessel_low_pass(
        #[case] order: usize,
        #[case] cutoff_freq: f64,
        #[case] expected_num: Vec<f64>,
        #[case] expected_den: Vec<f64>,
        #[case] epsilon: f64,
    ) {
        let tf = design_bessel(order, cutoff_freq, BandType::LowPass);
        assert_relative_eq!(tf.num, DVector::from_vec(expected_num), epsilon = epsilon);
        assert_relative_eq!(tf.den, DVector::from_vec(expected_den), epsilon = epsilon);
    }

    #[rstest]
    #[case(0, vec![1.0])]
    #[case(1, vec![1.0, 0.0])]
    #[case(2, vec![2.0, 0.0, -1.0])]
    #[case(3, vec![4.0, 0.0, -3.0, 0.0])]
    #[case(4, vec![8.0, 0.0, -8.0, 0.0, 1.0])]
    fn test_chebyshev1_polynomial(#[case] order: usize, #[case] expected_poly: Vec<f64>) {
        let poly = chebyshev1_polynomial(order);
        assert_relative_eq!(poly, DVector::from_vec(expected_poly));
    }

    #[rstest]
    #[case(1, 100.0, 1.0, vec![196.52267283602717], vec![1.0, 196.52267283602717], 1e-15)]
    #[case(2, 100.0, 1.0, vec![9826.133641801356], vec![1.0, 109.77343285639276, 11025.103280538484], 1e-15)]
    #[case(3, 100.0, 1.0, vec![491306.6820900678], vec![1.0, 98.8341209884761, 12384.091735782364, 491306.6820900678], 1e-9)]
    #[case(4, 90.0, 0.1, vec![53736256.63180374], vec![1.0, 162.33952536509088, 21277.06074788149, 1476589.879470203, 54358493.15756986], 1e-7)]
    #[case(1, 100.0, 3.0, vec![100.23772930076005], vec![1.0, 100.23772930076005], 1e-15)]
    #[case(2, 100.0, 3.0, vec![5011.886465038001], vec![1.0, 64.48996513028668, 7079.477801252795], 1e-11)]
    #[case(3, 100.0, 3.0, vec![250594.32325190006], vec![1.0, 59.72404165413484, 9283.480575752415, 250594.32325190003], 1e-9)]
    fn test_chebyshev1_low_pass(
        #[case] order: usize,
        #[case] cutoff_freq: f64,
        #[case] ripple_db: f64,
        #[case] expected_num: Vec<f64>,
        #[case] expected_den: Vec<f64>,
        #[case] epsilon: f64,
    ) {
        let tf = design_chebyshev1(order, cutoff_freq, ripple_db, BandType::LowPass);
        assert_relative_eq!(tf.num, DVector::from_vec(expected_num), epsilon = epsilon);
        assert_relative_eq!(tf.den, DVector::from_vec(expected_den), epsilon = epsilon);
    }

    #[rstest]
    #[case(1, 100.0, 1.0, vec![1.0, 0.0], vec![1.0, 50.88471399095875], 1e-15)]
    #[case(2, 100.0, 1.0, vec![0.8912509381337455, 0.0, 0.0], vec![1.0, 99.56680682544251, 9070.20981622186], 1e-15)]
    #[case(3, 100.0, 1.0, vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 252.0643864052326, 20116.58391618567, 2035388.559638349], 1e-8)]
    #[case(4, 100.0, 1.0, vec![0.8912509381337455, 0.0, 0.0, 0.0, 0.0], vec![1.0, 269.4285411067784, 52749.61060352098, 3456879.650283327, 362808392.6488744], 1e-14)]
    #[case(1, 100.0, 3.0, vec![1.0, 0.0], vec![1.0, 99.76283451109836], 1e-13)]
    #[case(2, 100.0, 3.0, vec![0.7079457843841378, 0.0, 0.0], vec![1.0, 91.09424019787792, 14125.335626068898], 1e-11)]
    #[case(3, 100.0, 3.0, vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 370.45853454631384, 23832.9587355016, 3990513.3804439357], 1e-8)]
    fn test_chebyshev1_high_pass(
        #[case] order: usize,
        #[case] cutoff_freq: f64,
        #[case] ripple_db: f64,
        #[case] expected_num: Vec<f64>,
        #[case] expected_den: Vec<f64>,
        #[case] epsilon: f64,
    ) {
        let tf = design_chebyshev1(order, cutoff_freq, ripple_db, BandType::HighPass);
        assert_relative_eq!(tf.num, DVector::from_vec(expected_num), epsilon = epsilon);
        assert_relative_eq!(tf.den, DVector::from_vec(expected_den), epsilon = epsilon);
    }

    #[rstest]
    #[case(1, 100.0, 1.0, vec![196.52267283602717], vec![1.0, 196.52267283602717], 1e-15)]
    #[case(2, 100.0, 1.0, vec![0.8912509381337451, 0.0, 17825.018762674903], vec![1.0, 62.26482262384244, 17825.01876267491], 1e-10)]
    #[case(3, 100.0, 1.0, vec![589.5680185080812, 0.0, 7860906.913441084], vec![1.0, 631.729847643468, 25746.0759780469, 7860906.913441085], 1e-8)]
    fn test_chebyshev2_low_pass(
        #[case] order: usize,
        #[case] cutoff_freq: f64,
        #[case] ripple_db: f64,
        #[case] expected_num: Vec<f64>,
        #[case] expected_den: Vec<f64>,
        #[case] epsilon: f64,
    ) {
        let tf = design_chebyshev2(order, cutoff_freq, ripple_db, BandType::LowPass);
        assert_relative_eq!(tf.num, DVector::from_vec(expected_num), epsilon = epsilon);
        assert_relative_eq!(tf.den, DVector::from_vec(expected_den), epsilon = epsilon);
    }

    #[rstest]
    #[case(1, 100.0, 1.0, vec![1.0, 0.0], vec![1.0, 50.88471399], 1e-9)]
    #[case(2, 100.0, 1.0, vec![1.0, 0.0, 5000.0], vec![1.0, 34.931140018894816, 5610.0922715098195], 1e-15)]
    #[case(3, 100.0, 1.0, vec![1.0, 0.0, 7500.0, 0.0], vec![1.0, 32.752042813310254, 8036.348154222454, 127211.78497739685], 1e-13)]
    #[case(3, 100.0, 2.0, vec![1.0, 0.0, 7500.0, 0.0], vec![1.0, 47.42911405686058, 8624.760430109342, 191195.77539480195], 1e-13)]
    fn test_chebyshev2_high_pass(
        #[case] order: usize,
        #[case] cutoff_freq: f64,
        #[case] ripple_db: f64,
        #[case] expected_num: Vec<f64>,
        #[case] expected_den: Vec<f64>,
        #[case] epsilon: f64,
    ) {
        let tf = design_chebyshev2(order, cutoff_freq, ripple_db, BandType::HighPass);
        assert_relative_eq!(tf.num, DVector::from_vec(expected_num), epsilon = epsilon);
        assert_relative_eq!(tf.den, DVector::from_vec(expected_den), epsilon = epsilon);
    }
}
