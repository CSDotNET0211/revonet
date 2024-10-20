use rand::{Rng};
use std;

use ea::*;

/// Represents baseline interface for the objective function.
///
/// By default solution is represented by a vector of real-numbers.
///
/// # Example: Custom optimization problem
/// ```
/// extern crate revonet;
/// extern crate rand;
///
/// use rand::{Rng, SeedableRng, StdRng};
///
/// use revonet::ea::*;
/// use revonet::problem::*;
///
/// // Dummy problem returning random fitness.
/// pub struct DummyProblem;
///
/// impl Problem for DummyProblem {
///     // Function to evaluate a specific individual.
///     fn compute<T: Individual>(&self, ind: &mut T) -> f32 {
///         // use `to_vec` to get real-coded representation of an individual.
///         let v = ind.to_vec().unwrap();
///
///         // Perform calculations as per optimization problem being implemened.
///         // Here just a random value is returned.
///         let mut rng: StdRng = StdRng::from_seed(&[0]);
///         rng.gen::<f32>()
///     }
/// }
///
/// fn main() {}
/// ```

pub trait Problem {
	/// Returns whether given fitness value is enough to be a solution.
	///
	/// # Arguments:
	/// * `value` - fitness value to consider.
	fn is_solution(&self, value: f32) -> bool {
		value < 1e-3f32
	}
	/// Generate random individual for the problem. Default implementation creates a real-coded
	/// individual with the number of genes equal to `size`
	///
	/// # Arguments:
	/// * `size` - number of genes.
	/// * `rng` - reference to pre-initialized RNG.
	fn get_random_individual<T: Individual, R: Rng>(&self, size: usize, mut rng: &mut R) -> T {
		let mut res_ind = T::new();
		res_ind.init(size, rng);
		res_ind
	}
	/// Computes fitness value for a given individual.
	///
	/// # Arguments:
	/// * `ind` - individual to compute a fitness for.
	fn compute<T: Individual>(&self, ind: &mut T) -> f32;
	fn compute_battle<T: Individual>(&self, ind1: &mut T, ind2: &mut T) -> (f32, f32);
}


//---------------------------------------------------------------
/*
Some other functions: https://en.wikipedia.org/wiki/Test_functions_for_optimization
*/

/// Sample problem representing [Sphere function](https://en.wikipedia.org/wiki/Test_functions_for_optimization).
#[allow(dead_code)]
#[derive(Clone)]
pub struct SphereProblem;

impl Problem for SphereProblem {
	fn compute<T: Individual>(&self, ind: &mut T) -> f32 {
		let v = ind.to_vec().expect("Can not extract vector of genes");
		if v.len() > 0 { v.iter().fold(0f32, |s, x| s + x * x) } else { std::f32::NAN }
	}

	fn compute_battle<T: Individual>(&self, ind1: &mut T, ind2: &mut T) -> (f32, f32) {
		panic!()
	}
}

//---------------------------------------------------------------
/// Sample problem representing [Rastrigin function](https://en.wikipedia.org/wiki/Rastrigin_function).
#[allow(dead_code)]
pub struct RastriginProblem;

impl Problem for RastriginProblem {
	fn compute<T: Individual>(&self, ind: &mut T) -> f32 {
		const PI2: f32 = 2f32 * std::f32::consts::PI;
		let v = ind.to_vec().expect("Can not extract vector of genes");
		if v.len() > 0 { v.iter().fold(10f32 * (v.len() as f32), |s, x| s + x * x - 10f32 * (x * PI2).cos()) } else { std::f32::NAN }
	}

	fn compute_battle<T: Individual>(&self, ind1: &mut T, ind2: &mut T) -> (f32, f32) {
		panic!()
	}
}


//---------------------------------------------------------------
/// Sample problem representing [Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function).
#[allow(dead_code)]
pub struct RosenbrockProblem;

impl Problem for RosenbrockProblem {
	fn compute<T: Individual>(&self, ind: &mut T) -> f32 {
		let v = ind.to_vec().expect("Can not extract vector of genes");
		if v.len() == 0 { return std::f32::NAN; }

		let mut res = 0f32;
		for k in 1..v.len() {
			let xk = v[k as usize];
			let xk1 = v[(k - 1) as usize];
			res += 100f32 * (xk - xk1).powf(2f32) + (xk1 - 1f32).powf(2f32)
		}
		res
	}

	fn compute_battle<T: Individual>(&self, ind1: &mut T, ind2: &mut T) -> (f32, f32) {
		panic!()
	}
}
