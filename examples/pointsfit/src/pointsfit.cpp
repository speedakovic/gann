#include <gann/gann.hpp>

#include <cmath>
#include <cstdlib>

#include <chrono>
#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <functional>

static const std::vector<std::vector<double>> POINTS = {{-1.5, -1.5}, {-0.5, 0}, {0, -1}, {0.5, 0}, {1.5, 1.5}};

static const std::vector<size_t> MLP_ARCH = {1, 4, 4, 1};
static const std::vector<gann::mlp::activation_function> MLP_AF = {gann::mlp::af_logistic, gann::mlp::af_logistic, gann::mlp::af_identity};

static const std::vector<double> SINGLE_LIMITS = {-10, 10, 0};

static const size_t POPSIZE  = 100;
static const size_t ELISIZE  = 5;
static const size_t GENMAX   = 10000;
static const size_t CONVN    = 1000;
static const double CONVNMAX = .99;
static const double SCOREMAX = 10000.;//std::nan("");

static const char* STATS_FILENAME = "pointsfit_stats.txt";
static const char* FUNC_FILENAME = "pointsfit_func.txt";

static double eval(const gann::mlp &p, double x)
{
	return p.propagate({x})[0];
}

static void evaluator_single(const std::vector<double> &params, double &score)
{
	gann::mlp p;
	p.set_architecture(MLP_ARCH);
	p.set_activation_functions_by_layers(MLP_AF);
	p.set_weights(params);

	double diff = 0.;
	for (const auto & point : POINTS)
		diff += pow(eval(p, point[0]) - point[1], 2);

	score = 1 / gann::nz(diff);
}

static void evaluator_multi(const std::vector<std::vector<double>> &params, std::vector<double> &scores)
{
	gann::mlp p;
	p.set_architecture(MLP_ARCH);
	p.set_activation_functions_by_layers(MLP_AF);

	for (size_t i = 0; i < params.size(); ++i) {

		p.set_weights(params[i]);

		double diff = 0.;
		for (const auto & point : POINTS)
			diff += pow(eval(p, point[0]) - point[1], 2);

		scores[i] = 1 / gann::nz(diff);
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

static std::vector<std::vector<double>> ga_limits(const std::vector<size_t> &arch, const std::vector<double> & single_limits)
{
	size_t limits_size = 0;
	size_t inputs = arch[0];
	for (size_t layer = 1; layer < arch.size(); ++layer) {
		limits_size += arch[layer] * (inputs + 1);
		inputs = arch[layer];
	}

	std::vector<std::vector<double>> limits;
	for (size_t i = 0; i < limits_size; ++i)
		limits.push_back(single_limits);

	return limits;
}

static void draw_func(const std::vector<double> & params)
{
	gann::mlp p;
	p.set_architecture(MLP_ARCH);
	p.set_activation_functions_by_layers(MLP_AF);
	p.set_weights(params);

	const double  step = 0.01;
	const double  min  = -2;
	const double  max  = +2;
	std::ofstream file(FUNC_FILENAME, std::ios::out);

	file << std::fixed << std::setprecision(6);
	for (double x = min; x <= max; x += step)
		file << x << '\t' << eval(p, x) << std::endl;
	file << std::endl << std::endl;

	for (const auto & point : POINTS)
		file << point[0] << '\t' << point[1] << std::endl;
}

int main()
{
	using gann::operator<<;
	using namespace std::placeholders;

	std::fstream stats_file(STATS_FILENAME, std::ios::out);

	gann::selection_op_tournament          selection(2, 10);
	gann::crossover_op_multiple_arithmetic crossover;
	gann::mutation_op_normal               mutation(0.5);
	gann::score_scaler_none                scaler;

	std::vector<double> best_params;
	double best_score;

	gann::ga_simple ga(ga_limits(MLP_ARCH, SINGLE_LIMITS), selection, crossover, mutation, scaler,
	                   POPSIZE, ELISIZE, GENMAX, CONVN, CONVNMAX, SCOREMAX, 0);

	auto begin = std::chrono::steady_clock::now();
	ga(evaluator_multi, std::bind(statistics_listener, std::ref(stats_file), _1), best_params, best_score);
	auto end = std::chrono::steady_clock::now();

	draw_func(best_params);

	std::cout << "duration [us] : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
	std::cout << "best params   : " << best_params << std::endl;
	std::cout << "best score    : " << best_score << std::endl;

	return EXIT_SUCCESS;
}

