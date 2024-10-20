use rand::{Rng};
use std;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use chrono::Local;

use context::*;
use ea::*;
use ga::*;
use math::*;
use neuro::*;
use problem::*;
use result::*;
use settings::*;


/// Represents individual for neuroevolution. The main difference is that the NE individual also
/// has a `network` field, which stores current neural network.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NEIndividual {
	genes: Vec<f32>,
	fitness: f32,
	network: Option<MultilayeredNetwork>,
}

#[allow(dead_code)]
impl<'a> NEIndividual {
	/// Update individual's network by assigning gene values to network's weights.
	fn update_net(&mut self) {
		// println!("update net");
		match &mut self.network {
			&mut Some(ref mut net) => {
				// copy weights from genes to NN.
				let (mut ws, mut bs) = (Vec::new(), Vec::new());
				// println!("{:?}", self.genes);
				let mut cur_idx = 0;
				// weights.
				for layer in net.iter_layers() {
					// println!("getting slice for weights {}..{}", cur_idx, (cur_idx + inputs_num * layer.len()));
					let inputs_num = layer.get_inputs_num();
					ws.push(Vec::from(&self.genes[cur_idx..(cur_idx + inputs_num * layer.len())]));
					cur_idx += inputs_num * layer.len();
					// inputs_num = layer.len();
				}
				// biases.
				for layer in net.iter_layers() {
					// println!("getting slice for biases {}..{}", cur_idx, (cur_idx + layer.len()));
					bs.push(Vec::from(&self.genes[cur_idx..(cur_idx + layer.len())]));
					cur_idx += layer.len();
				}
				// println!("set weights");
				net.set_weights(&ws, &bs);
			}
			&mut None => {
				println!("[update_net] Warning: network is not defined");
			}
		}
	}
}

impl Individual for NEIndividual {
	fn new() -> NEIndividual {
		NEIndividual {
			genes: Vec::new(),
			fitness: std::f32::NAN,
			network: None,
		}
	}

	fn init<R: Rng>(&mut self, size: usize, mut rng: &mut R) {
		self.genes = rand_vector_std_gauss(size as usize, rng);
	}

	fn get_fitness(&self) -> f32 {
		self.fitness
	}

	fn set_fitness(&mut self, fitness: f32) {
		self.fitness = fitness;
	}

	fn to_vec(&self) -> Option<&[f32]> {
		Some(&self.genes)
	}

	fn to_vec_mut(&mut self) -> Option<&mut Vec<f32>> {
		Some(&mut self.genes)
	}

	fn to_net(&mut self) -> Option<&MultilayeredNetwork> {
		self.update_net();
		match &self.network {
			&Some(ref net) => Some(net),
			&None => None
		}
	}

	fn to_net_mut(&mut self) -> Option<&mut MultilayeredNetwork> {
		self.update_net();
		match &mut self.network {
			&mut Some(ref mut net) => { Some(net) }
			&mut None => None
		}
	}

	fn set_net(&mut self, net: MultilayeredNetwork) {
		let (ws, bs) = net.get_weights();
		self.network = Some(net);
		self.genes = ws.into_iter()
			.fold(Vec::new(), |mut res, w| {
				res.extend(w.iter().cloned());
				res
			});
		self.genes.extend(bs.into_iter()
			.fold(Vec::new(), |mut res, b| {
				res.extend(b.iter().cloned());
				res
			}));
	}
}

//================================================================================

/// Structure for neuroevolutionary algorithm.
///
/// # Example: Run neuroevolutionary algorithm to solve XOR problem.
/// ```
/// extern crate revonet;
///
/// use revonet::ea::*;
/// use revonet::ne::*;
/// use revonet::neproblem::*;
/// use revonet::settings::*;
///
/// fn main() {
///     let (pop_size, gen_count, param_count) = (20, 50, 100); // gene_count does not matter here as NN structure is defined by a problem.
///     let settings = EASettings::new(pop_size, gen_count, param_count);
///     let problem = XorProblem::new();
///
///     let mut ne: NE<XorProblem> = NE::new(&problem);
///     let res = ne.run(settings).expect("Error: NE result is empty");
///     println!("result: {:?}", res);
///     println!("\nbest individual: {:?}", res.best);
/// }
/// ```

pub struct NE<'a, P: Problem + 'a> {
	/// Context structure containing information about GA run, its progress and results.
	ctx: Option<EAContext<NEIndividual>>,
	/// Reference to the objective function object implementing `Problem` trait.
	problem: &'a P,
}

#[allow(dead_code)]
impl<'a, P: Problem> NE<'a, P> {
	/// Create a new neuroevolutionary algorithm for the given problem.
	pub fn new(problem: &'a P) -> NE<'a, P> {
		NE {
			problem: problem,
			ctx: None,
		}
	}
}

impl<'a, P: Problem + Sync + std::marker::Send + 'static + std::clone::Clone> EA<'a, P> for NE<'a, P> {
	type IndType = NEIndividual;

	fn breed(&self, ctx: &mut EAContext<Self::IndType>, sel_inds: &Vec<usize>, children: &mut Vec<Self::IndType>) {
		cross(&ctx.population, sel_inds, children, ctx.settings.use_elite, ctx.settings.x_type, ctx.settings.x_prob, ctx.settings.x_alpha, &mut ctx.rng);
		mutate(children, ctx.settings.mut_type, ctx.settings.mut_prob, ctx.settings.mut_sigma, &mut ctx.rng);
	}

	fn run(&mut self, settings: EASettings, battle_learning: &bool) -> Result<&EAResult<Self::IndType>, ()> {
		println!("run");

		/*let path;
		{
			let date = Local::now();
			let date_str = date.format("%Y_%m_%d_%H:%M").to_string();
			let json = serde_json::to_string(&settings).unwrap();
			path = format!("backup/{date_str}");
			let raw_path = format!("{path}/settings.txt");
			let relative_path = Path::new(&raw_path);

			if let Some(parent) = relative_path.parent() {
				std::fs::create_dir_all(parent).unwrap();
			}
			let mut file = File::create(relative_path).unwrap();
			file.write_all(json.as_bytes()).unwrap();
		}*/

		let gen_count = settings.gen_count;
		let mut ctx = EAContext::new(settings, self.problem);
		self.run_with_context(&mut ctx, self.problem, gen_count, battle_learning);
		self.ctx = Some(ctx);
		Ok(&(&self.ctx.as_ref().expect("Empty EAContext")).result)
	}
}

//===================================================================

#[cfg(test)]
#[allow(unused_imports)]
mod test {
	use rand;

	use math::*;
	use ne::*;
	use neproblem::*;

	#[test]
	pub fn test_symbolic_regression() {
		let (pop_size, gen_count, param_count) = (20, 20, 100);
		let settings = EASettings::new(pop_size, gen_count, param_count);
		let problem = SymbolicRegressionProblem::new_f();

		let mut ne: NE<SymbolicRegressionProblem> = NE::new(&problem);
		let res = ne.run(settings, &false).expect("Error: NE result is empty");
		println!("result: {:?}", res);
		println!("\nbest individual: {:?}", res.best);
		// println!("\nbest individual NN: {:?}", res.best.to_net());
		// let ne = NE::new(&problem);
	}

	#[test]
	pub fn test_net_get_set() {
		let mut rng = rand::thread_rng();
		let mut net = MultilayeredNetwork::new(2, 2);
		net.add_hidden_layer(10 as usize, ActivationFunctionType::Sigmoid)
			.add_hidden_layer(5 as usize, ActivationFunctionType::Sigmoid)
			.build(&mut rng, NeuralArchitecture::Multilayered);
		let (ws1, bs1) = net.get_weights();

		let mut ind = NEIndividual::new();
		ind.set_net(net.clone());
		let net2 = ind.to_net_mut().unwrap();
		let (ws2, bs2) = net2.get_weights();

		// compare ws1 & ws2 and bs1 & bs2. Should be equal.
		for k in 0..ws1.len() {
			let max_diff = max(&sub(&ws1[k], &ws2[k]));
			println!("max diff: {}", max_diff);
			assert!(max_diff == 0f32);

			let max_diff = max(&sub(&bs1[k], &bs2[k]));
			println!("bs1: {:?}", bs1[k]);
			println!("bs2: {:?}", bs2[k]);
			println!("max diff: {}", max_diff);
			assert!(max_diff == 0f32);
		}
	}
}
