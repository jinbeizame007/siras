use criterion::*;
use nalgebra::{Complex, DVector};
use rand::Rng;
use siras::fft::{fft, ifft};

fn fft_benchmark(c: &mut Criterion) {
    let n = 1024;
    let mut rng = rand::thread_rng();
    let signal = DVector::<f64>::from_fn(n, |_, _| rng.gen());
    c.bench_function("fft", |b| b.iter(|| fft(black_box(&signal))));
}

fn ifft_benchmark(c: &mut Criterion) {
    let n = 1024;
    let mut rng = rand::thread_rng();
    let spectrum = DVector::<Complex<f64>>::from_fn(n, |_, _| Complex::new(rng.gen(), rng.gen()));
    c.bench_function("ifft", |b| b.iter(|| ifft(black_box(&spectrum))));
}

criterion_group!(benches, fft_benchmark, ifft_benchmark);

criterion_main!(benches);
