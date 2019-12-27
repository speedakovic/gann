#include <gann/gann.hpp>

#include <cmath>
#include <cstdlib>

#include <chrono>
#include <vector>
#include <iostream>

static const double X = 1.;
static const double Y = 2.;

static const size_t POPSIZE  = 50;
static const size_t ELISIZE  = 1;
static const size_t GENMAX   = 500;
static const size_t CONVN    = 50;
static const double CONVNMAX = .99;
static const double SCOREMAX = std::nan("");

void evaluator_single(const std::vector<double> &params, double &score)
{
	double x = params[0];
	double y = params[1];
	double z = pow(x - X, 2) + pow(y - Y, 2);
	score = 1 / gann::nz(z);
}

void evaluator_multi(const std::vector<std::vector<double>> &params, std::vector<double> &scores)
{
	for (size_t i = 0; i < params.size(); ++i) {
		double x = params[i][0];
		double y = params[i][1];
		double z = pow(x - X, 2) + pow(y - Y, 2);
		scores[i] = 1 / gann::nz(z);
	}
}

void statistics_listener(const gann::ga_simple::statistics &stats)
{
	std::cout << "gen: " << stats.generation
	          << ", best: " << stats.best_score << ", worst: " << stats.worst_score
	          << ", mean: " << stats.mean_score << ", median: " << stats.median_score
	          << ", conv: " << stats.convergence << std::endl;
}

int main()
{
	using gann::operator<<;

	gann::selection_op_roulette            selection(10);
	gann::crossover_op_multiple_arithmetic crossover;
	gann::mutation_op_normal               mutation(0.2);
	gann::score_scaler_linear_nz           scaler;

	std::vector<double> best_params;
	double best_score;

	gann::ga_simple ga({{-10, 10, 0}, {-10, 10, 0}}, selection, crossover, mutation, scaler,
	                   POPSIZE, ELISIZE, GENMAX, CONVN, CONVNMAX, SCOREMAX, 0);

	auto begin = std::chrono::steady_clock::now();
	ga(evaluator_multi, statistics_listener, best_params, best_score);
	auto end = std::chrono::steady_clock::now();

	std::cout << "duration [us]   : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
	std::cout << "expected params : " << std::vector<double>{X,Y} << std::endl;
	std::cout << "best params     : " << best_params              << std::endl;
	std::cout << "best score      : " << best_score               << std::endl;

	return EXIT_SUCCESS;
}

