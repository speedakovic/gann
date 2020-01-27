#include <gann/gann.hpp>

#include <cmath>
#include <cstdlib>

#include <chrono>
#include <vector>
#include <fstream>
#include <iostream>
#include <functional>

static const double X = 1.;
static const double Y = 2.;

static const size_t POPSIZE  = 50;
static const size_t ELISIZE  = 1;
static const size_t GENMAX   = 500;
static const size_t CONVN    = 50;
static const double CONVNMAX = .99;
static const double SCOREMAX = std::nan("");

static const char* STATS_FILENAME = "paraboloid_stats.txt";

static void evaluator_single(const std::vector<double> &params, double &score)
{
	double x = params[0];
	double y = params[1];
	double z = pow(x - X, 2) + pow(y - Y, 2);
	score = 1 / gann::nz(z);
}

static void evaluator_multi(const std::vector<std::vector<double>> &params, std::vector<double> &scores)
{
	for (size_t i = 0; i < params.size(); ++i) {
		double x = params[i][0];
		double y = params[i][1];
		double z = pow(x - X, 2) + pow(y - Y, 2);
		scores[i] = 1 / gann::nz(z);
	}
}

static void statistics_listener(std::fstream &stats_file, const gann::ga_simple::statistics &stats)
{
	std::cout << "gen: " << stats.generation
	          << ", best: " << stats.best_score << ", worst: " << stats.worst_score
	          << ", mean: " << stats.mean_score << ", median: " << stats.median_score
	          << ", conv: " << stats.convergence << std::endl;

	stats_file << stats.generation << "\t"
	           << 1 / gann::nz(stats.best_score) << "\t"
	           << 1 / gann::nz(stats.mean_score) << "\t"
	           << 1 / gann::nz(stats.median_score) << std::endl;
}

int main()
{
	using gann::operator<<;
	using namespace std::placeholders;

	std::fstream stats_file(STATS_FILENAME, std::ios::out);

	gann::selection_op_roulette       selection(10);
	gann::crossover_op_arithmetic_all crossover;
	gann::mutation_op_normal_single   mutation(0.2);
	gann::score_scaler_linear_nz      scaler;

	std::vector<double> best_params;
	double best_score;

	gann::ga_simple ga({{-10, 10, 0}, {-10, 10, 0}}, selection, crossover, mutation, scaler,
	                   POPSIZE, ELISIZE, GENMAX, CONVN, CONVNMAX, SCOREMAX, 0);

	auto begin = std::chrono::steady_clock::now();
	ga(evaluator_multi, std::bind(statistics_listener, std::ref(stats_file), _1), best_params, best_score);
	auto end = std::chrono::steady_clock::now();

	std::cout << "duration [us]   : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
	std::cout << "expected params : " << std::vector<double>{X,Y} << std::endl;
	std::cout << "best params     : " << best_params              << std::endl;
	std::cout << "best score      : " << best_score               << std::endl;

	return EXIT_SUCCESS;
}

