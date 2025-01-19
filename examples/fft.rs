use std::f64::consts::PI;
use std::fs;

use nalgebra::{ComplexField, DVector};
use plotters::prelude::*;

extern crate siras;
use siras::fft::{fft, fftfreq, ifft};

fn plot(
    x: &DVector<f64>,
    y: &DVector<f64>,
    (w, h): (u32, u32),
    path: &str,
    title: &str,
    x_label: &str,
    y_label: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (w, h)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30, FontStyle::Normal).into_font())
        .margin(20)
        .x_label_area_size(60)
        .y_label_area_size(70)
        .build_cartesian_2d(x.min()..x.max(), y.min()..(y.max() * 1.1))?;

    let label_font_x = ("sans-serif", 25, FontStyle::Normal).into_font();
    let label_font_y = ("sans-serif", 25, FontStyle::Normal).into_font();
    chart
        .configure_mesh()
        .x_desc(x_label)
        .y_desc(y_label)
        .x_label_style(label_font_x)
        .y_label_style(label_font_y)
        .draw()?;

    chart
        .draw_series(LineSeries::new(
            x.iter().copied().zip(y.iter().copied()),
            Palette99::pick(3).stroke_width(2),
        ))?
        .label(title)
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], Palette99::pick(3)));

    root.present()?;

    Ok(())
}

fn main() {
    let frequency0 = 10.0;
    let frequency1 = 100.0;
    let amplitude0 = 1.0;
    let amplitude1 = 0.5;
    let sample_frequency = 1024;
    let dt = 1.0 / sample_frequency as f64;
    let t = DVector::from_iterator(
        sample_frequency,
        (0..sample_frequency).map(|i| i as f64 / sample_frequency as f64),
    );
    let signal = (2.0 * PI * frequency0 * t.clone()).map(|e| amplitude0 * e.sin())
        + (2.0 * PI * frequency1 * t.clone()).map(|e| amplitude1 * e.sin());

    let spectrums = fft(&signal);
    let reconstructed_signal = ifft(&spectrums).map(|e| e.re);

    let amplitudes = spectrums
        .rows(0, 200)
        .map(|e| e.abs() / (sample_frequency as f64 / 2.0));
    let freqencies: DVector<f64> = fftfreq(t.len(), dt).rows(0, 200).into();

    let plot_dir = "examples/plots";
    if !std::path::Path::new(plot_dir).exists() {
        fs::create_dir_all(plot_dir).unwrap();
    }

    plot(
        &t,
        &signal,
        (1200, 600),
        &format!("{}/original_signal.png", plot_dir),
        "original signal",
        "time",
        "amplitude",
    )
    .unwrap();

    plot(
        &freqencies,
        &amplitudes,
        (1200, 600),
        &format!("{}/fft.png", plot_dir),
        "fft",
        "frequency",
        "amplitude",
    )
    .unwrap();

    plot(
        &t,
        &reconstructed_signal,
        (1200, 600),
        &format!("{}/reconstructed_signal.png", plot_dir),
        "reconstructed signal",
        "time",
        "amplitude",
    )
    .unwrap();
}
