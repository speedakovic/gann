#include <gann/gann.h>

#include <cmath>
#include <cstdlib>

#include <vector>
#include <iostream>

static double X = 1.;
static double Y = 2.;

static size_t POPSIZE  = 50;
static size_t ELISIZE  = 1;
static size_t GENMAX   = 500;
static size_t CONVN    = 50;
static double CONVNMAX = .99;

class paraboloid_ev : public gann::evaluator
{
public:
	virtual bool run(const std::vector<double> &params, double &score) const
	{
		double x = params[0];
		double y = params[1];
		double z = pow(x - X, 2) + pow(y - Y, 2);
		score = 1 / z;
		return true;
	};
};

int main()
{
	using gann::operator<<;

	gann::ga                               ga;
	gann::selection_op_roulette            selection;
	gann::crossover_op_multiple_arithmetic crossover;
	gann::mutation_op_normal               mutation(0.2);
	gann::score_scaler_linear              scaler;

	paraboloid_ev ev;

	std::vector<double> best_params;
	double best_score;

	if (!ga.configure({{-10, 10, 0}, {-10, 10, 0}}, &selection, &crossover, &mutation, &scaler, POPSIZE, ELISIZE, GENMAX, CONVN, CONVNMAX, 0)) {
		std::cerr << "configuring genetic algorithm failed" << std::endl;
		return EXIT_FAILURE;
	}

	if (!ga.run(ev, best_params, best_score)) {
		std::cerr << "running genetic algorithm failed" << std::endl;
		return EXIT_FAILURE;
	}

	std::cout << "expected params : " << std::vector<double>{X,Y} << std::endl;
	std::cout << "best params     : " << best_params              << std::endl;
	std::cout << "best score      : " << best_score               << std::endl;

	return EXIT_SUCCESS;
}

