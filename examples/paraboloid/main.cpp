#include <gann/gann.h>

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

bool evaluator_single(const std::vector<double> &params, double &score)
{
	double x = params[0];
	double y = params[1];
	double z = pow(x - X, 2) + pow(y - Y, 2);
	score = 1 / z;
	return true;
};

bool evaluator_multi(const std::vector<std::vector<double>> &params, std::vector<double> &scores)
{
	for (size_t i = 0; i < params.size(); ++i) {
		double x = params[i][0];
		double y = params[i][1];
		double z = pow(x - X, 2) + pow(y - Y, 2);
		scores[i] = 1 / z;
	}
	return true;
};

int main()
{
	using gann::operator<<;

	gann::selection_op_roulette            selection(10);
	gann::crossover_op_multiple_arithmetic crossover;
	gann::mutation_op_normal               mutation(0.2);
	gann::score_scaler_linear              scaler;

	std::vector<double> best_params;
	double best_score;

	gann::ga_simple ga({{-10, 10, 0}, {-10, 10, 0}}, selection, crossover, mutation, scaler,
	                   POPSIZE, ELISIZE, GENMAX, CONVN, CONVNMAX, SCOREMAX, 0);

	auto begin = std::chrono::steady_clock::now();
	if (!ga.run(evaluator_multi, best_params, best_score)) {
		std::cerr << "running genetic algorithm failed" << std::endl;
		return EXIT_FAILURE;
	}
	auto end = std::chrono::steady_clock::now();

	std::cout << "duration [us]   : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
	std::cout << "expected params : " << std::vector<double>{X,Y} << std::endl;
	std::cout << "best params     : " << best_params              << std::endl;
	std::cout << "best score      : " << best_score               << std::endl;

	return EXIT_SUCCESS;
}

