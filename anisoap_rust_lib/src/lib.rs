mod calc_utils;
mod ellip_expansion;

use chemfiles::{Frame, Trajectory};
use ellip_expansion::compute_moments::compute_moments_rust;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::{exceptions::PyAssertionError, prelude::*};

#[pymodule]
fn anisoap_rust_lib(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn compute_moments<'py>(
        py: Python<'py>,
        mat: PyReadonlyArray2<'_, f64>,
        g_vec: PyReadonlyArray1<'_, f64>,
        max_deg: i32,
    ) -> PyResult<&'py PyArray3<f64>> {
        match compute_moments_rust(mat.as_array(), g_vec.as_array(), max_deg) {
            Ok(moment_array) => return Ok(moment_array.into_pyarray(py)),
            Err(message) => return Err(PyErr::new::<PyAssertionError, String>(message)),
        }
    }

    #[pyfn(m)]
    fn ellipsoid_transform<'py>(_py: Python<'py>, file_path: String) -> () {
        let mut file = Trajectory::open(file_path, 'r').expect("Failed to open.");
        let frames: Vec<Frame> = (0..file.nsteps())
            .map(|frame_index: usize| {
                let mut frame = Frame::new();
                file.read_step(frame_index, &mut frame).unwrap();
                frame
            })
            .collect();

        fn print_f64_2d(pos_vec: &[[f64; 3]]) -> String {
            let mut result = "[[".to_string();
            result.push_str(
                &pos_vec
                    .iter()
                    .map(|pos| {
                        pos.iter()
                            .map(|val| format!("{:.4}", val))
                            .collect::<Vec<String>>()
                            .join(", ")
                    })
                    .collect::<Vec<String>>()
                    .join("], ["),
            );
            result.push_str("]]");
            result
        }

        for (frame_index, frame) in frames.iter().enumerate() {
            println!(
                "{:>4}: {} {}",
                frame_index,
                frame.atom(0).name(),
                print_f64_2d(frame.positions())
            )
        }
    }

    Ok(())
}
