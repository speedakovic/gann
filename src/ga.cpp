#include <gann/ga.h>
#include <gann/log.h>

#include <cmath>
#include <mutex>
#include <thread>
#include <random>
#include <numeric>
#include <iostream>
#include <algorithm>
#include <functional>

namespace gann
{

////////////////////////////////////////////////////////////////////////////////
// selection operators
////////////////////////////////////////////////////////////////////////////////

void selection_op_roulette::run(const std::vector<double> &scores, std::vector<std::vector<double>> &population) const
{
	double s = std::accumulate(scores.begin(), scores.end(), 0.);

	if (!std::isnormal(s) || !std::isgreater(s, 0.)) {
		GANN_ERR("unexpected scores, selection by copy" << std::endl);
		return;
	}

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> distr(0, s);

	std::vector<std::vector<double>> population_old(population);

	for (size_t i = 0; i < population.size(); ++i) {

		double r = s - distr(mt);
		size_t j = 0;

		for (; r > 0. && j < scores.size(); ++j)
			r -= scores[j];

		population[i] = population_old[j - 1];
	}
}

////////////////////////////////////////////////////////////////////////////////
// crossover operators
////////////////////////////////////////////////////////////////////////////////

void crossover_op_single_arithmetic::run(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<size_t> distr_index(0, limits.size() - 1);
	std::uniform_real_distribution<double> distr_alpha(0, 1);

	for (size_t i = 0; i < population.size() - 1; i += 2) {

		std::vector<double> &ind1 = population[i];
		std::vector<double> &ind2 = population[i + 1];

		size_t index = distr_index(mt);
		double alpha = distr_alpha(mt);

		double ind1_par = ind1[index];
		double ind2_par = ind2[index];

		ind1[index] = ind1_par * alpha	+ ind2_par * (1 - alpha);
		ind2[index] = ind2_par * alpha	+ ind1_par * (1 - alpha);
	}
}

void crossover_op_multiple_arithmetic::run(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<size_t> distr_index(0, limits.size() - 1);
	std::uniform_real_distribution<double> distr_alpha(0, 1);

	for (size_t i = 0; i < population.size() - 1; i += 2) {

		std::vector<double> &ind1 = population[i];
		std::vector<double> &ind2 = population[i + 1];

		std::vector<size_t> indexes(limits.size());
		std::iota(indexes.begin(), indexes.end(), 0);
		std::shuffle(indexes.begin(), indexes.end(), mt);
		indexes.resize(distr_index(mt));

		for (const auto &index : indexes) {

			double alpha = distr_alpha(mt);

			double ind1_par = ind1[index];
			double ind2_par = ind2[index];

			ind1[index] = ind1_par * alpha	+ ind2_par * (1 - alpha);
			ind2[index] = ind2_par * alpha	+ ind1_par * (1 - alpha);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// mutation operators
////////////////////////////////////////////////////////////////////////////////

void mutation_op_uniform::run(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<size_t> distr_pop_index(0, population.size() / p - 1);
	std::uniform_int_distribution<size_t> distr_param_index(0, limits.size() - 1);
	std::vector<std::uniform_real_distribution<double>> distr_param;

	for (const auto &limit : limits)
		distr_param.push_back(std::uniform_real_distribution<double>(limit[0], limit[1]));

	for (auto &ind : population) {
		size_t pop_index = distr_pop_index(mt);
		if (pop_index < population.size()) {
			size_t param_index = distr_param_index(mt);
			ind[param_index] = distr_param[param_index](mt);
		}
	}
}

void mutation_op_normal::run(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<size_t> distr_pop_index(0, population.size() / p - 1);
	std::uniform_int_distribution<size_t> distr_param_index(0, limits.size() - 1);
	std::vector<std::normal_distribution<double>> distr_param;

	for (const auto &limit : limits)
		distr_param.push_back(std::normal_distribution<double>(0., c * (limit[1] - limit[0])));

	for (auto &ind : population) {
		size_t pop_index = distr_pop_index(mt);
		if (pop_index < population.size()) {
			size_t param_index = distr_param_index(mt);
			double param = distr_param[param_index](mt);
			if (!static_cast<int>(limits[param_index][2])) {
				if (param < limits[param_index][0])
					param = limits[param_index][0];
				else if (param > limits[param_index][1])
					param = limits[param_index][1];
			}
			ind[param_index] = param;
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// score scalers
////////////////////////////////////////////////////////////////////////////////

void score_scaler_none::run(const std::vector<double> &scores, std::vector<double> &scores_scaled) const
{
	std::copy(scores.begin(), scores.end(), scores_scaled.begin());
}

void score_scaler_offset::run(const std::vector<double> &scores, std::vector<double> &scores_scaled) const
{
	double tmp = *std::min_element(scores.begin(), scores.end());
	std::transform(scores.begin(), scores.end(), scores_scaled.begin(), [&tmp](const double &score){return score - tmp;});
}

void score_scaler_linear::run(const std::vector<double> &scores, std::vector<double> &scores_scaled) const
{
	double tmp = *std::min_element(scores.begin(), scores.end());
	std::transform(scores.begin(), scores.end(), scores_scaled.begin(), [&tmp](const double &score){return score - tmp;});

	tmp = *std::max_element(scores_scaled.begin(), scores_scaled.end());
	if (std::isnormal(tmp))
		std::transform(scores_scaled.begin(), scores_scaled.end(), scores_scaled.begin(), [&tmp](const double &score){return score / tmp;});
}

////////////////////////////////////////////////////////////////////////////////
// genetic algorithms
////////////////////////////////////////////////////////////////////////////////

ga_simple::ga_simple() :
	limits	  (),
	selection (),
	crossover (),
	mutation  (),
	scaler	  (),
	popsize   (),
	elisize   (),
	genmax	  (),
	convn	  (),
	convmax   (),
	thnum	  ()
{
}

bool ga_simple::configure(const std::vector<std::vector<double>> &limits,
				   const selection_op *selection, const crossover_op *crossover, const mutation_op *mutation, const score_scaler *scaler,
				   const size_t &popsize, const size_t &elisize, const size_t &genmax, const size_t &convn, const double &convmax, const size_t &thnum)
{
	this->limits = limits;

	this->selection = selection;
	this->crossover = crossover;
	this->mutation	= mutation;
	this->scaler	= scaler;

	this->popsize = popsize % 2 ? popsize + 1: popsize;
	this->elisize = elisize;

	this->genmax  = genmax;
	this->convn   = convn;
	this->convmax = convmax;

	this->thnum = thnum > 0 ? thnum : std::thread::hardware_concurrency();

	return true;
}

// conv = bestscore[current - convn] / bestscore[current]
bool ga_simple::run(const evaluator_single &eval, std::vector<double> &params, double &score) const
{
	std::vector<std::vector<double>> population(popsize, std::vector<double>(limits.size()));
	std::vector<std::vector<double>> elite(elisize, std::vector<double>(limits.size()));
	std::vector<double> scores_scaled(popsize);
	std::vector<double> scores(popsize);
	std::queue<double> best_scores;

	std::vector<size_t> i_scores(popsize);
	double mean_score;
	double median_score;
	double conv;

	size_t gencnt = 0;

	GANN_DBG("running simple ga with single-evaluator..." << std::endl);
	GANN_DBG("running threads: " << thnum << std::endl);

	if (!initialize_population(population)) {
		GANN_ERR("initializing population failed" << std::endl);
		return false;
	}

	if (!calculate_scores_mt(eval, population, scores, scores_scaled)) {
		GANN_ERR("calculating scores failed" << std::endl);
		return false;
	}

	++gencnt;

	calculate_stats(scores, i_scores, mean_score, median_score);
	calculate_convergence(conv, best_scores, scores[i_scores[0]]);

	GANN_DBG("gen: " << gencnt << ", best: " << scores[i_scores[0]] << ", mean: " << mean_score << ", median: " << median_score << ", conv: " << conv << std::endl);

	while (true) {

		if (genmax > 0 && gencnt >= genmax) {
			GANN_DBG("maximum number of generations reached" << std::endl);
			break;
		}

		if (std::isnormal(conv) && conv >= convmax) {
			GANN_DBG("maximum convergence reached" << std::endl);
			break;
		}

		for (size_t i = 0; i < elisize; ++i)
			elite[i] = population[i_scores[i]];

		selection->run(scores_scaled, population);
		crossover->run(limits, population);
		mutation ->run(limits, population);

		for (size_t i = 0; i < elisize; ++i)
			population[i] = elite[i];

		if (!calculate_scores_mt(eval, population, scores, scores_scaled)) {
			GANN_ERR("calculating scores failed" << std::endl);
			return false;
		}

		++gencnt;

		calculate_stats(scores, i_scores, mean_score, median_score);
		calculate_convergence(conv, best_scores, scores[i_scores[0]]);
		
		GANN_DBG("gen: " << gencnt << ", best: " << scores[i_scores[0]] << ", mean: " << mean_score << ", median: " << median_score << ", conv: " << conv << std::endl);
	}

	params = population[i_scores[0]];
	score  = scores[i_scores[0]];

	return true;
}

// conv = bestscore[current - convn] / bestscore[current]
bool ga_simple::run(const evaluator_multi &eval, std::vector<double> &params, double &score) const
{
	std::vector<std::vector<double>> population(popsize, std::vector<double>(limits.size()));
	std::vector<std::vector<double>> elite(elisize, std::vector<double>(limits.size()));
	std::vector<double> scores_scaled(popsize);
	std::vector<double> scores(popsize);
	std::queue<double> best_scores;

	std::vector<size_t> i_scores(popsize);
	double mean_score;
	double median_score;
	double conv;

	size_t gencnt = 0;

	GANN_DBG("running simple ga with multi-evaluator..." << std::endl);

	if (!initialize_population(population)) {
		GANN_ERR("initializing population failed" << std::endl);
		return false;
	}

	if (!eval.run(population, scores)) {
		GANN_ERR("evaluating scores failed" << std::endl);
		return false;
	}

	scaler->run(scores, scores_scaled);

	++gencnt;

	calculate_stats(scores, i_scores, mean_score, median_score);
	calculate_convergence(conv, best_scores, scores[i_scores[0]]);

	GANN_DBG("gen: " << gencnt << ", best: " << scores[i_scores[0]] << ", mean: " << mean_score << ", median: " << median_score << ", conv: " << conv << std::endl);

	while (true) {

		if (genmax > 0 && gencnt >= genmax) {
			GANN_DBG("maximum number of generations reached" << std::endl);
			break;
		}

		if (std::isnormal(conv) && conv >= convmax) {
			GANN_DBG("maximum convergence reached" << std::endl);
			break;
		}

		for (size_t i = 0; i < elisize; ++i)
			elite[i] = population[i_scores[i]];

		selection->run(scores_scaled, population);
		crossover->run(limits, population);
		mutation ->run(limits, population);

		for (size_t i = 0; i < elisize; ++i)
			population[i] = elite[i];

		if (!eval.run(population, scores)) {
			GANN_ERR("evaluating scores failed" << std::endl);
			return false;
		}

		scaler->run(scores, scores_scaled);

		++gencnt;

		calculate_stats(scores, i_scores, mean_score, median_score);
		calculate_convergence(conv, best_scores, scores[i_scores[0]]);

		GANN_DBG("gen: " << gencnt << ", best: " << scores[i_scores[0]] << ", mean: " << mean_score << ", median: " << median_score << ", conv: " << conv << std::endl);
	}

	params = population[i_scores[0]];
	score  = scores[i_scores[0]];

	return true;
}

bool ga_simple::initialize_population(std::vector<std::vector<double>> &population) const
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::vector<std::uniform_real_distribution<double>> distr;

	for (const auto &limit : limits)
		distr.push_back(std::uniform_real_distribution<double>(limit[0], limit[1]));

	for (size_t i = 0; i < population.size(); ++i)
		for (size_t j = 0; j < population[i].size(); ++j)
			population[i][j] = distr[j](mt); // population[i][j] = (limits[j][0] + limits[j][1]) / 2;

	return true;
}

bool ga_simple::calculate_scores(const evaluator_single &eval, const std::vector<std::vector<double>> &population, std::vector<double> &scores, std::vector<double> &scores_scaled) const
{
	for (size_t i = 0; i < population.size(); ++i) {

		double score;

		if (!eval.run(population[i], score)) {
			GANN_ERR("running evaluator failed" << std::endl);
			return false;
		}

		scores[i] = score;
	}

	scaler->run(scores, scores_scaled);

	return true;
}

static void evaluator_runner(const evaluator_single &eval, std::mutex &mutex, const std::vector<std::vector<double>> &population, std::vector<double> &scores, size_t &index, int &err)
{
	size_t i;
	double score;
	bool iserr = false;

	mutex.lock();

	for (;;) {

		if (index >= population.size()) {
			mutex.unlock();
			break;
		}

		i = index++;

		mutex.unlock();

		if (!eval.run(population[i], score)) {
			GANN_ERR("evaluator runner " << std::this_thread::get_id() << " failed because of running evaluator failed" << std::endl);
			iserr = true;
			break;
		}

		mutex.lock();

		scores[i] = score;
	}

	if (iserr) {
		mutex.lock();
		err = 1;
		mutex.unlock();
	}
}

bool ga_simple::calculate_scores_mt(const evaluator_single &eval, const std::vector<std::vector<double>> &population, std::vector<double> &scores, std::vector<double> &scores_scaled) const
{
	int err = 0;
	size_t index = 0;
	std::mutex mutex;
	std::vector<std::thread> threads;

	for (size_t i = 0; i < thnum; ++i)
		threads.push_back(std::thread(evaluator_runner, std::cref(eval),
						  std::ref(mutex), std::cref(population), std::ref(scores), std::ref(index), std::ref(err)));

	for(auto &thread : threads)
		thread.join();

	if (err)
		return false;

	scaler->run(scores, scores_scaled);

	return true;
}

void ga_simple::calculate_stats(const std::vector<double> &scores, std::vector<size_t> &i_scores, double &mean_score, double &median_score) const
{
	std::iota(i_scores.begin(), i_scores.end(), static_cast<size_t>(0));
	std::sort(i_scores.begin(), i_scores.end(), [&scores](const size_t &x, const size_t &y){return scores[x] > scores[y];});

	mean_score = std::accumulate(scores.begin(), scores.end(), 0.) / scores.size();
	median_score = scores[i_scores[i_scores.size() / 2]];
}

void ga_simple::calculate_convergence(double &conv, std::queue<double> &best_scores, const double &best_score) const
{
	if (0 == convn) {
		conv = std::nan("");
	} else if (best_scores.size() < convn) {
		conv = std::nan("");
		best_scores.push(best_score);
	} else {
		conv = best_scores.front() / best_score;
		best_scores.pop();
		best_scores.push(best_score);
	}
}

} // namespace gann

