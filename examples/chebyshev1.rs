use std::f64::consts::PI;
use std::fs;

use nalgebra::DVector;
use plotters::prelude::*;

extern crate siras;
use siras::filter_design::{chebyshev1, digital_to_analog_cutoff, FilterType};

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
        .build_cartesian_2d(x.min()..x.max(), -2.0..2.0)?;

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
    let f0 = 10.0;
    let f1 = 100.0;
    let sample_frequency = 32000;
    let order = 4;
    let ripple = 0.1;
    let t = DVector::from_iterator(
        sample_frequency,
        (0..sample_frequency).map(|i| i as f64 / sample_frequency as f64),
    );
    let x =
        (2.0 * PI * f0 * t.clone()).map(|e| e.sin()) + (2.0 * PI * f1 * t.clone()).map(|e| e.sin());

    let cutoff_frequency_high_pass = digital_to_analog_cutoff(98.0, sample_frequency as f64);
    let cutoff_frequency_low_pass = digital_to_analog_cutoff(15.0, sample_frequency as f64);

    let alpha = 0.5;
    let signal_with_high_pass_filter = chebyshev1(
        order,
        cutoff_frequency_high_pass,
        ripple,
        FilterType::HighPass,
    )
    .to_discrete(1.0 / sample_frequency as f64, alpha)
    .filtfilt(&x, &t);
    let signal_with_low_pass_filter = chebyshev1(
        order,
        cutoff_frequency_low_pass,
        ripple,
        FilterType::LowPass,
    )
    .to_discrete(1.0 / sample_frequency as f64, alpha)
    .filtfilt(&x, &t);

    let plot_dir = "examples/plots";
    if !std::path::Path::new(plot_dir).exists() {
        fs::create_dir_all(plot_dir).unwrap();
    }

    plot(
        &t,
        &x,
        (1200, 600),
        &format!("{}/chebyshev1_without_filter.png", plot_dir),
        "without filter",
        "time",
        "amplitude",
    )
    .unwrap();
    plot(
        &t,
        &signal_with_low_pass_filter,
        (1200, 600),
        &format!("{}/chebyshev1_with_low_pass_filter.png", plot_dir),
        "with low pass filter",
        "time",
        "amplitude",
    )
    .unwrap();
    plot(
        &t,
        &signal_with_high_pass_filter,
        (1200, 600),
        &format!("{}/chebyshev1_with_high_pass_filter.png", plot_dir),
        "with high pass filter",
        "time",
        "amplitude",
    )
    .unwrap();
}
