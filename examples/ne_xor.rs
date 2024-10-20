extern crate revonet;
extern crate serde;

use std::time::Instant;
use serde::Serialize;
use revonet::ea::*;
use revonet::ne::*;
use revonet::neproblem::*;
use revonet::neuro::{ActivationFunctionType, MultilayeredNetwork, NeuralArchitecture, NeuralNetwork};
use revonet::result::EAResult;
use revonet::settings::*;

fn main() {
	let (pop_size, gen_count, param_count) = (50, 500, 100); // gene_count does not matter here as NN structure is defined by a problem.
	let settings = EASettings::new(pop_size, gen_count, param_count);
	let problem = XorProblem::new();

	let mut ne: NE<XorProblem> = NE::new(&problem);
	let res = ne.run(settings, &false).expect("Error: NE result is empty");
	println!("result: {:?}", res);
	println!("\nbest individual: {:?}", res.best);

	let mut rng = rand::thread_rng();
	let mut net: MultilayeredNetwork = MultilayeredNetwork::new(25, 1);
	net.add_hidden_layer(10 as usize, ActivationFunctionType::Relu)
		.add_hidden_layer(2 as usize, ActivationFunctionType::Relu)
		.build(&mut rng, NeuralArchitecture::Multilayered);
	// .build(&mut rng, NeuralArchitecture::BypassInputs);

	let mut best = res.best.clone();
	let net = best.to_net_mut().unwrap();
	//net.serialize(());
	//std::io::stdin().read_line(&mut "".to_string());
	let data = [1f32; 10];

	let start = Instant::now();
	for _ in 0..10_000 {
		net.compute(&data);
	}
	let duration = start.elapsed();
	println!("time:{:?}", duration);
	return;
	// 10ビットの全パターン数は2^10 = 1024
	let total_patterns = 1 << 10;

	for bit_pattern in 0..total_patterns {
		// ビットパターンを配列にコピー
		let mut bit_array = [0f32; 10];
		for i in 0..10 {
			bit_array[i] = ((bit_pattern >> i) & 1) as f32;
		}

		// 1の数を数える
		let count_ones = bit_array.iter().filter(|&&bit| bit == 1f32).count();

		// 偶数なら1、奇数なら0を評価
		let result = if count_ones % 2 == 0 { 1 } else { 0 };

		let result = net.compute(&bit_array);
		println!("{:?} => {}", bit_array, result[0]);
	}
}
