use rand::{Rng};
use rand::distributions::{IndependentSample, Range};
use serde::de::{DeserializeOwned};
use serde::ser::Serialize;
use std;
use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::atomic::Ordering::SeqCst;
use chrono::Local;

use context::*;
use math::*;
use neuro::{MultilayeredNetwork};
use problem::*;
use result::*;
use settings::*;

use rayon::prelude::*;
use ne::NEIndividual;

/// Trait representing functionality required to evolve an individual for optimization
/// and NN tuning tasks.
///
/// Contains functions to retrieve genes or neural network from an individual and get/set its fitness.
#[allow(dead_code, unused_variables)]
pub trait Individual {
	/// Creates a new individual with empty set of genes and NAN fitness.
	fn new() -> Self;
	/// Initializes an individual by allocating random vector of genes using Gaussian distribution.
	///
	/// # Arguments
	/// * `size` - number of genes.
	/// * `rng` - mutable reference to the external RNG.
	fn init<R: Rng>(&mut self, size: usize, &mut R);
	/// Return current fitness value.
	fn get_fitness(&self) -> f32;
	/// Update fitness value.
	fn set_fitness(&mut self, fitness: f32);
	/// Return vector of genes.
	fn to_vec(&self) -> Option<&[f32]>;
	/// Return mutable vector of genes.
	fn to_vec_mut(&mut self) -> Option<&mut Vec<f32>>;
	/// Return `MultilayeredNetwork` object with weights assigned according to the genes' values.
	fn to_net(&mut self) -> Option<&MultilayeredNetwork> { None }
	/// Return mutable `MultilayeredNetwork` object with weights assigned according to the genes' values.
	fn to_net_mut(&mut self) -> Option<&mut MultilayeredNetwork> { None }
	/// Update individual's `MultilayeredNetwork` object and update genes according to the network weights.
	///
	/// # Arguments:
	/// * `net` - neural network to update from.
	fn set_net(&mut self, net: MultilayeredNetwork) {}
}

/// Represents real-coded individual with genes encoded as vector of real numbers.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RealCodedIndividual {
	/// Collection of individual genes.
	pub genes: Vec<f32>,
	/// Fitness value associated with the individual.
	pub fitness: f32,
}

impl RealCodedIndividual {}

impl Individual for RealCodedIndividual {
	fn new() -> Self {
		RealCodedIndividual { genes: Vec::new(), fitness: std::f32::NAN }
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
}

//======================================================================

/// Trait for an evolutionary algorithm.
/// Defines functions which are typical for running a common EA.
/// To implement a trait a function `breed` should be implemented.
pub trait EA<'a, P> where P: Problem + Sync + Send + Clone + 'static, {
	type IndType: Individual + Clone + Serialize + DeserializeOwned + Send + Sync + 'static;
	/*fn battle(mut ind1: Box<Individual>, mut ind2: Self::IndType, problem: Arc<P>, ind1_win_count: &mut Arc<AtomicUsize>, ind2_win_count: &mut Arc<AtomicUsize>) {
		rayon::spawn(move || {
			let f = problem.compute_battle(&mut ind1, &mut ind2);
		
		});
	}*/
	fn run_multiple(&mut self, settings: EASettings, run_num: u32) -> Result<EAResultMultiple<Self::IndType>, ()> {
		let run_ress = (0..run_num).into_iter()
			.map(|_| {
				self.run(settings.clone(), &false).expect("Error during GA run").clone()
			})
			.collect::<Vec<EAResult<Self::IndType>>>();
		let res = EAResultMultiple::new(&run_ress);
		Ok(res)
	}

	/// "Main" function for the EA which runs a cycle for an evolutionary search.
	///
	/// # Arguments:
	/// * `ctx` - `EAContext` object containing information regarding current EA run.
	/// * `problem` - reference to the `Problem` trait which specifies an objective function.
	/// * `gen_count` - number of generations (iterations) for search.
	fn run_with_context(&self, ctx: &mut EAContext<Self::IndType>, problem: &P, gen_count: u32, battle_learning: &bool) { // -> Result<Rc<&'a EAResult<T>>, ()> {
		// let mut ctx = self.get_context_mut();
		// println!("run_with_context");
		for t in 0..gen_count {
			// evaluation
			// println!("evaluation");
			self.evaluate(ctx, problem, battle_learning);

			// selection
			// println!("selection");
			let sel_inds = self.select(ctx);

			// crossover
			// println!("crossover");
			let mut children: Vec<Self::IndType> = Vec::with_capacity(ctx.settings.pop_size as usize);
			self.breed(ctx, &sel_inds, &mut children);

			// next gen
			// println!("next_generation");
			self.next_generation(ctx, &children);

			let mut fitness_clone = ctx.fitness.clone();
			fitness_clone.sort_by(|a, b| b.partial_cmp(a).unwrap());
			println!("> {} : {:?}", t, &fitness_clone);
			println!(" Best fitness at generation {} : {}\n", t, min(&ctx.fitness));

			let best_index = ctx.fitness
				.iter()
				.enumerate()
				.min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
				.map(|(index, &fitness)| (index, fitness)).unwrap().0;

			ctx.fitness.iter().skip(best_index).take(1).for_each(|eval| {
				dbg!(eval);
			});

			ctx.population.iter().skip(best_index).take(1).for_each(|ind| {
				if t % 10 == 0 {
					let json = serde_json::to_string(&ind.clone().to_net()).unwrap();
					let raw_path = format!("genes/gen_{t}.txt");
					let relative_path = Path::new(&raw_path);

					if let Some(parent) = relative_path.parent() {
						std::fs::create_dir_all(parent).unwrap();
					}

					let mut file = File::create(relative_path).unwrap();

					file.write_all(json.as_bytes()).unwrap();
				}
			});

			/*	let _ = ctx.population.iter().map(|ind| {
					let mut indivisual = NEIndividual::new();
					indivisual.
				});
	
				{
					let json = serde_json::to_string(&ctx.population).unwrap();
					let raw_path = format!("{path}/gen_{t}.txt");
					let relative_path = Path::new(&raw_path);
	
					if let Some(parent) = relative_path.parent() {
						std::fs::create_dir_all(parent).unwrap();
					}
					let mut file = File::create(relative_path).unwrap();
					file.write_all(json.as_bytes()).unwrap();
				}*/
		}

		// Ok(Rc::new(&ctx.result))
	}

	/// Function to evaluate current population for the given `problem`. In result of evaluation
	///   fitness for every individual is updated as well as statistics regarding mean, min, max
	///   fitness values.
	///
	/// # Arguments:
	/// * `ctx` - `EAContext` object containing information regarding current EA run.
	/// * `problem` - reference to the `Problem` trait which specifies an objective function.
	fn evaluate(&self, ctx: &mut EAContext<Self::IndType>, problem: &P, battle_learning: &bool) {
		let cur_result = &mut ctx.result;
		let mut popul: &mut Vec<_> = &mut ctx.population;
		//ctx.settings.

		if *battle_learning {
			//対戦する人一覧
			let mut battle_list: Vec<_> = popul.iter_mut().collect();

			for player in &mut battle_list {
				player.set_fitness(0.);
			}

			let mut rng = rand::thread_rng();

			//トーナメント、対戦する人が１になるまで
			loop {
				let battle_list_len = battle_list.len();

				if battle_list_len == 1 {
					break;
				}

				//対戦相手をシャッフル
				for i in (1..battle_list_len).rev() {
					let j = rng.gen_range(0, i + 1); // 0からiの間の乱数を生成
					battle_list.swap(i, j); // 要素を入れ替え
				}

				//順番に対戦カードを作成
				let mut battle_card: Vec<_> = Vec::new();
				for _ in 0..(battle_list_len / 2) {
					battle_card.push((battle_list.remove(0), battle_list.remove(0)));
				}

				//マルチスレッドで試合
				let winners = battle_card.into_par_iter().map(|(ind1, ind2)| {
					//	let a = ind1.clone();
					//battle_cardを回して10先、Battle構造体で
					let mut battle = Battle {
						ind1: ind1.clone(),
						ind2: ind2.clone(),
						problem: problem.clone(),
						ind1_win_count: Arc::new(AtomicUsize::new(0)),
						ind2_win_count: Arc::new(AtomicUsize::new(0)),
						total_battle_count: Arc::new(AtomicUsize::new(0)),
						first_to: 10,
					};
					battle.compute();


					let result: (f32, f32);

					let test1 = battle.ind1_win_count.load(SeqCst);
					let test2 = battle.ind2_win_count.load(SeqCst);

					if !(test1 == battle.first_to || test2 == battle.first_to) {
						panic!("aaa");
					}

					if battle.ind1_win_count.load(SeqCst) > battle.ind2_win_count.load(SeqCst) {
						result = (-2., 0.);
					} else {
						result = (0., -2.);
					}

					if ind1.get_fitness().is_nan() {
						ind1.set_fitness(0.);
					}
					if ind2.get_fitness().is_nan() {
						ind2.set_fitness(0.);
					}

					let ind1_fitness = ind1.get_fitness() + result.0;
					let ind2_fitness = ind2.get_fitness() + result.1;

					ind1.set_fitness(ind1_fitness);
					ind2.set_fitness(ind2_fitness);

					//勝者
					if ind1_fitness < ind2_fitness {
						ind1
					} else {
						ind2
					}
				}).collect();


				battle_list = winners;
				//	dbg!(battle_list.len());
			}

			ctx.fitness = popul.iter().map(|ref ind| {
				ind.get_fitness()
			}).collect::<Vec<f32>>();
		} else {
			ctx.fitness = popul.par_iter_mut().map(|ref mut ind| {
				let f = problem.compute(ind as &mut Self::IndType);
				ind.set_fitness(f);
				f
			}).collect::<Vec<f32>>();
		}

		let fits = &ctx.fitness;
		// println!("{:?}", fits);
		/*if cur_result.first_hit_fe_count == 0 {
			for k in 0..fits.len() {
				if problem.is_solution(fits[k]) {
					cur_result.first_hit_fe_count = cur_result.fe_count + (k + 1) as u32;
					break;
				}
			}
		}*/

		cur_result.avg_fitness.push(mean(&fits));
		cur_result.max_fitness.push(max(&fits));

		let last_min_fitness = min(&fits);
		cur_result.min_fitness.push(last_min_fitness);
		if cur_result.best.get_fitness().is_nan() || (cur_result.best.get_fitness() > last_min_fitness) {
			let idx = (&fits).iter().position(|&x| x == last_min_fitness).expect("Min fitness is not found");
			cur_result.best = popul[idx].clone();
			cur_result.best_fe_count = cur_result.fe_count + (idx + 1) as u32;
		}
		cur_result.best.set_fitness(last_min_fitness);
		cur_result.fe_count += fits.len() as u32;
	}

	/// Function to select individuals for breeding. Updates given `ctx`.
	///
	/// # Arguments:
	/// * `ctx` - `EAContext` object containing information regarding current EA run.
	fn select(&self, ctx: &mut EAContext<Self::IndType>) -> Vec<usize> {
		select_tournament(&ctx.fitness, ctx.settings.tour_size, &mut ctx.rng)
	}

	/// Function to prepare population for the next generation. By default copies children obtained
	///   in result of `breed` to the `ctx.population`.
	///
	/// # Arguments:
	/// * `ctx` - `EAContext` object containing information regarding current EA run.
	/// * `children` - reference to the vector of children individuals which should
	///                get to the next generation.
	fn next_generation(&self, ctx: &mut EAContext<Self::IndType>, children: &Vec<Self::IndType>) {
		ctx.population = Vec::with_capacity(children.len());
		for k in 0..children.len() {
			ctx.population.push(children[k].clone());
		}
		// ctx.population = children.iter().map(|ref c| c.clone()).collect::<Vec<T>>();
		ctx.population.truncate(ctx.settings.pop_size as usize);
	}

	/// Function to create children individuals using current context and selected individuals.
	///
	/// # Arguments:
	/// * `ctx` - `EAContext` object containing information regarding current EA run.
	/// * `sel_inds` - vector of indices of individuals from `ctx.population` selected for breeding.
	/// * `children` - reference to the container to store resulting children individuals.
	fn breed(&self, ctx: &mut EAContext<Self::IndType>, sel_inds: &Vec<usize>, children: &mut Vec<Self::IndType>);

	/// Run evolutionary algorithm and return `EAResult` object.
	///
	/// # Arguments:
	/// * `settings` - `EASettings` object.
	// fn run(&mut self, settings: EASettings) -> Result<&EAResult<Self::IndType>, ()>;
	fn run(&mut self, settings: EASettings, battle_learning: &bool) -> Result<&EAResult<Self::IndType>, ()>;
}

struct Battle<T: Individual + Clone + Send, P: Problem + Sync> {
	pub ind1: T,
	pub ind2: T,
	pub problem: P,
	pub ind1_win_count: Arc<AtomicUsize>,
	pub ind2_win_count: Arc<AtomicUsize>,
	pub total_battle_count: Arc<AtomicUsize>,
	pub first_to: usize,
}

impl<T: Individual + Clone + Send + 'static + Sync, P: Problem + Sync + Clone + Send + 'static> Battle<T, P> {
	pub fn compute(&mut self) {
		let mut loop_count = Arc::new(Mutex::new(self.first_to));

		'label: while {
			let mut extra = Arc::new(Mutex::new(0));

			rayon::scope(|scope| {
				self.total_battle_count.fetch_add(*loop_count.lock().unwrap(), SeqCst);

				while *loop_count.lock().unwrap() > 0 {
					let ind1 = self.ind1.clone();
					let ind2 = self.ind2.clone();
					let problem = self.problem.clone();
					let ind1_win_count = self.ind1_win_count.clone();
					let ind2_win_count = self.ind2_win_count.clone();
					let first_to = self.first_to;
					let total_battle_count = self.total_battle_count.clone();

					//		let extra = Arc::new(Mutex::new(0usize));
					let extra_clone = extra.clone();
					scope.spawn(move |_| {
						Self::internal(ind1, ind2, problem, ind1_win_count, ind2_win_count, first_to, total_battle_count, extra_clone);
					});
					//		*loop_count.lock().unwrap() += *extra.lock().unwrap();

					*loop_count.lock().unwrap() -= 1;
				}
			});
			//		println!("extra:{}", *extra.lock().unwrap());

			*loop_count.lock().unwrap() = *extra.lock().unwrap();
			//		println!("loop_count:{}", *loop_count.lock().unwrap());

			if self.ind1_win_count.load(SeqCst) + self.ind2_win_count.load(SeqCst) > self.first_to * 2 {
				println!("30回以上試合することはありえない、強制終了。{} vs {}", self.ind1_win_count.load(SeqCst), self.ind2_win_count.load(SeqCst));
				break 'label;
			}

			*loop_count.lock().unwrap() != 0
		} {}
	}

	fn internal(mut ind1: T, mut ind2: T, problem: P, ind1_win_count: Arc<AtomicUsize>, ind2_win_count: Arc<AtomicUsize>, first_to: usize, total_battle_count: Arc<AtomicUsize>, extra: Arc<Mutex<usize>>) {
		//	println!("ゲーム開始");

		let f = problem.compute_battle(&mut ind1, &mut ind2);
		assert_eq!(f.0 + f.1, 1.);

		ind1_win_count.fetch_add(f.0 as usize, SeqCst);
		ind2_win_count.fetch_add(f.1 as usize, SeqCst);
		//dbg!(f);


		let ind1_win = ind1_win_count.load(SeqCst);
		let ind2_win = ind2_win_count.load(SeqCst);
		let total_battle = total_battle_count.load(SeqCst);

		/*if ind1_win == first_to || ind2_win == first_to {
			println!("a");
		}*/
		//	println!("{} vs {} :{}", ind1_win, ind2_win, total_battle);


		if ind1_win + ind2_win == total_battle {
			if ind1_win == first_to || ind2_win == first_to {
				//		println!("終わり");
				//終わり
			} else {

				//終わった数と予定の数が同じ試合が全部終わった時、まだftに達してなかったら

				//1じゃなくても、first_to - デカいほうでいける
				let game_needed;
				if ind1_win > ind2_win {
					game_needed = first_to.saturating_sub(ind1_win);
					if (first_to as isize - ind1_win as isize) < 0 {
						println!("ゲームやりすぎ？ {} vs {}", first_to, ind1_win);
					}
				} else {
					game_needed = first_to.saturating_sub(ind2_win);
					if (first_to as isize - ind2_win as isize) < 0 {
						println!("ゲームやりすぎ？ {} vs {}", first_to, ind2_win);
					}
				}


				//	println!("足りないからまだ生成:{}", game_needed);
				//	println!("ind1:{}, ind2:{}, total:{}", ind1_win, ind2_win, total_battle);

				*extra.lock().unwrap() += game_needed;
			}
		}
	}
}

/// Creates population of given size. Uses `problem.get_random_individual` to generate a
/// new individual
///
/// # Arguments:
/// * `pop_size` - population size.
/// * `ind_size` - default size (number of genes) of individuals.
/// * `rng` - reference to pre-initialized RNG.
/// * `problem` - reference to the object implementing `Problem` trait.
pub fn create_population<T: Individual, P: Problem, R: Rng + Sized>(pop_size: u32, ind_size: u32, mut rng: &mut R, problem: &P) -> Vec<T> {
	println!("Creating population of {} individuals having size {}", pop_size, ind_size);
	let mut res = Vec::with_capacity(pop_size as usize);
	for _ in 0..pop_size {
		res.push(problem.get_random_individual::<T, R>(ind_size as usize, rng));
	}
	res
}

/// Implementation of the [tournament selection](https://en.wikipedia.org/wiki/Tournament_selection).
///
/// # Arguments:
/// * `fits` - fitness values. i-th element should be equal to the fitness of the i-th individual
///            in population.
/// * `tour_size` - tournament size. The bigger the higher is the selective pressure (more strict
///                 selection). Minimal acceptable value is 2.
/// * `rng` - reference to pre-initialized RNG.
fn select_tournament(fits: &Vec<f32>, tour_size: u32, mut rng: &mut Rng) -> Vec<usize> {
	let range = Range::new(0, fits.len());
	let mut sel_inds: Vec<usize> = Vec::with_capacity(fits.len());  // vector of indices of selected inds. +1 in case of elite RealCodedindividual is used.
	for _ in 0..fits.len() {
		let tour_inds = (0..tour_size).map(|_| range.ind_sample(&mut rng)).collect::<Vec<usize>>();
		let winner = tour_inds.iter().fold(tour_inds[0], |w_idx, &k|
			if fits[w_idx] < fits[k] { w_idx } else { k },
		);
		sel_inds.push(winner);
	}
	sel_inds
}

/// Get copy of the individual having the best fitness value.
///
/// # Arguments:
/// * `popul` - vector of individuals to select from.
pub fn get_best_individual<T: Individual + Clone>(popul: &Vec<T>) -> T {
	let min_fitness = popul.into_iter().fold(std::f32::MAX, |s, ref ind| if s < ind.get_fitness() { s } else { ind.get_fitness() });
	let idx = popul.into_iter().position(|ref x| x.get_fitness() == min_fitness).expect("Min fitness is not found");
	popul[idx].clone()
}

//========================================================

#[allow(unused_imports)]
#[cfg(test)]
mod test {
	use rand;

	use ea::*;
	use math::*;

	#[test]
	fn test_tournament_selection() {
		const IND_COUNT: usize = 100;
		const TRIAL_COUNT: u32 = 100;

		let mut prev_mean = 0.5f32;   // avg value in a random array in [0; 1].
		let mut rng = rand::thread_rng();
		for t in 2..10 {
			let mut cur_mean = 0f32;
			// compute mean fitness for the selected population for TRIAL_COUNT trials.
			for _ in 0..TRIAL_COUNT {
				let fitness_vals = rand_vector(IND_COUNT, &mut rng);
				let sel_inds = select_tournament(&fitness_vals, t, &mut rng);
				let tmp_mean = sel_inds.iter().fold(0f32, |s, &idx| s + fitness_vals[idx]) / IND_COUNT as f32;
				cur_mean += tmp_mean;
			}
			cur_mean /= TRIAL_COUNT as f32;
			// bigger tournaments should give smaller average fitness in selected population.
			assert!(cur_mean < prev_mean);
			prev_mean = cur_mean;
		}
	}
}