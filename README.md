# SIRAS: Signal Processing Library for Rust

<div align="center">
    <img src="media/siras.webp" alt="siras" width="45%">
</div>

```rust
use nalgebra::DVector;

extern crate siras;
use siras::filter_design::BandType;
use siras::lti::DiscreteTransferFunction;
use siras::signal_generator::generate_sine_wave;

fn main() {
    let sample_rate = 32000;
    let t = DVector::from_fn(sample_rate, |i, _| i as f64 / sample_rate as f64);

    let freq1 = 10.0;
    let freq2 = 100.0;
    let signal = generate_sine_wave(&t, freq1) + generate_sine_wave(&t, freq2);

    let order = 4;
    let dt = 1.0 / sample_rate as f64;
    let cutoff_freq = 20.0;

    let filter = DiscreteTransferFunction::butter(order, cutoff_freq, dt, BandType::LowPass);
    let filtered_signal = filter.filtfilt(&signal, &t);
}
```

# Features

## Filters

<details open>

<summary><b>Butterworth Filter</b></summary>

<div align="center">
    <img src="media/butter_without_filter.png" alt="siras">
    <img src="media/butter_with_low_pass_filter.png" alt="siras">
    <img src="media/butter_with_high_pass_filter.png" alt="siras">
</div>

</details>

<details>
<summary><b>Bessel Filter</b></summary>

<div align="center">
    <img src="media/bessel_without_filter.png" alt="siras">
    <img src="media/bessel_with_low_pass_filter.png" alt="siras">
    <img src="media/bessel_with_high_pass_filter.png" alt="siras">
</div>

</details>

<details>
<summary><b>Chebyshev Type I Filter</b></summary>

<div align="center">
    <img src="media/chebyshev1_without_filter.png" alt="siras">
    <img src="media/chebyshev1_with_low_pass_filter.png" alt="siras">
    <img src="media/chebyshev1_with_high_pass_filter.png" alt="siras">
</div>

</details>

<details>
<summary><b>Chebyshev Type II Filter</b></summary>

<div align="center">
    <img src="media/chebyshev2_without_filter.png" alt="siras">
    <img src="media/chebyshev2_with_low_pass_filter.png" alt="siras">
    <img src="media/chebyshev2_with_high_pass_filter.png" alt="siras">
</div>

</details>

## FFT

<details open>
<summary><b>FFT</b></summary>

<div align="center">
    <img src="media/original_signal.png" alt="siras">
</div>

### FFT (time domain -> frequency domain)

<div align="center">
    <img src="media/fft.png" alt="siras">
</div>

### IFFT (frequency domain -> time domain)

<div align="center">
    <img src="media/reconstructed_signal.png" alt="siras">
</div>

</details>
