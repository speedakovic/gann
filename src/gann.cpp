#include <gann/gann.hpp>

#include <thread>
#include <random>
#include <numeric>
#include <iostream>
#include <algorithm>

#ifdef NDEBUG
	#define GANN_DBG(x)
#else
	#define GANN_DBG(x) do {std::cerr << x;} while(0);
#endif

namespace gann
{

////////////////////////////////////////////////////////////////////////////////
// selection operators
////////////////////////////////////////////////////////////////////////////////

void selection_op_roulette::operator()(const std::vector<size_t> &i_scores, const std::vector<double> &scores, std::vector<std::vector<double>> &population) const
{
#if 1
	std::random_device rd;
	std::mt19937 mt(rd());
	std::discrete_distribution<size_t> distr(scores.begin(), scores.end());

	std::vector<std::vector<double>> population_old(population);

	for (size_t i = 0; i < population.size() - 1; i += 2) {

		size_t extra_run = extra_runs;
		size_t i0, i1;

		{
			i0 = distr(mt);
		}

		do {
			i1 = distr(mt);

		} while (population_old[i0] == population_old[i1] && extra_run--);

		population[i    ] = population_old[i0];
		population[i + 1] = population_old[i1];
	}

#else
	double s = std::accumulate(scores.begin(), scores.end(), 0.);

	checknormal(s, "not-normal number in roulette selection operator");
	checkgreater(s, 0., "not-greater than zero number in roulette selection operator");

	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> distr(0, s);

	std::vector<std::vector<double>> population_old(population);

	for (size_t i = 0; i < population.size() - 1; i += 2) {

		size_t extra_run = extra_runs;
		size_t i0, i1;

		{
			double r = s - distr(mt);
			size_t j = 0;

			for (; r > 0. && j < scores.size(); ++j)
				r -= scores[j];

			i0 = j - 1;
		}

		do {
			double r = s - distr(mt);
			size_t j = 0;

			for (; r > 0. && j < scores.size(); ++j)
				r -= scores[j];

			i1 = j - 1;

		} while (population_old[i0] == population_old[i1] && extra_run--);

		population[i    ] = population_old[i0];
		population[i + 1] = population_old[i1];
	}
#endif
}

void selection_op_rank::operator()(const std::vector<size_t> &i_scores, const std::vector<double> &scores, std::vector<std::vector<double>> &population) const
{
	std::vector<size_t> iscores(scores.size());
	for (size_t i = 0; i < scores.size(); ++i)
		iscores[i_scores[i]] = scores.size() - i;

	std::random_device rd;
	std::mt19937 mt(rd());
	std::discrete_distribution<size_t> distr(iscores.begin(), iscores.end());

	std::vector<std::vector<double>> population_old(population);

	for (size_t i = 0; i < population.size() - 1; i += 2) {

		size_t extra_run = extra_runs;
		size_t i0, i1;

		{
			i0 = distr(mt);
		}

		do {
			i1 = distr(mt);

		} while (population_old[i0] == population_old[i1] && extra_run--);

		population[i    ] = population_old[i0];
		population[i + 1] = population_old[i1];
	}
}

void selection_op_tournament::operator()(const std::vector<size_t> &i_scores, const std::vector<double> &scores, std::vector<std::vector<double>> &population) const
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<size_t> distr(0, population.size() - 1);

	std::vector<std::vector<double>> population_old(population);

	for (size_t i = 0; i < population.size() - 1; i += 2) {

		size_t extra_run = extra_runs;
		size_t i0, i1;

		{
			std::vector<size_t> competitors_i;
			std::vector<size_t> competitors_s;
			for (size_t j = 0; j < competitors_num; ++j) {
				size_t index = distr(mt);
				competitors_i.push_back(index);
				competitors_s.push_back(scores[index]);
			}
			i0 = competitors_i[std::distance(competitors_s.begin(), std::max_element(competitors_s.begin(), competitors_s.end()))];
		}

		{
			std::vector<size_t> competitors_i;
			std::vector<size_t> competitors_s;
			for (size_t j = 0; j < competitors_num; ++j) {
				size_t index = distr(mt);
				competitors_i.push_back(index);
				competitors_s.push_back(scores[index]);
			}
			i1 = competitors_i[std::distance(competitors_s.begin(), std::max_element(competitors_s.begin(), competitors_s.end()))];

		} while (population_old[i0] == population_old[i1] && extra_run--);

		population[i    ] = population_old[i0];
		population[i + 1] = population_old[i1];

	}
}

////////////////////////////////////////////////////////////////////////////////
// crossover operators
////////////////////////////////////////////////////////////////////////////////

void crossover_op_arithmetic_single::operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<size_t> distr_param_index(0, limits.size() - 1);
	std::uniform_real_distribution<double> distr_alpha(0, 1);

	for (size_t pop_index = 0; pop_index < population.size() - 1; pop_index += 2) {

		std::vector<double> &ind1 = population[pop_index];
		std::vector<double> &ind2 = population[pop_index + 1];

		size_t param_index = distr_param_index(mt);
		double alpha = distr_alpha(mt);

		double ind1_par = ind1[param_index];
		double ind2_par = ind2[param_index];

		ind1[param_index] = ind1_par * alpha + ind2_par * (1. - alpha);
		ind2[param_index] = ind2_par * alpha + ind1_par * (1. - alpha);

		if (!static_cast<int>(limits[param_index][2])) {
			if (ind1[param_index] < limits[param_index][0])
				ind1[param_index] = limits[param_index][0];
			else if (ind1[param_index] > limits[param_index][1])
				ind1[param_index] = limits[param_index][1];

			if (ind2[param_index] < limits[param_index][0])
				ind2[param_index] = limits[param_index][0];
			else if (ind2[param_index] > limits[param_index][1])
				ind2[param_index] = limits[param_index][1];
		}
	}
}

void crossover_op_arithmetic_all::operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> distr_alpha(0, 1);

	for (size_t pop_index = 0; pop_index < population.size() - 1; pop_index += 2) {

		std::vector<double> &ind1 = population[pop_index];
		std::vector<double> &ind2 = population[pop_index + 1];

		for (size_t param_index = 0; param_index < limits.size(); ++param_index) {

			double alpha = distr_alpha(mt);

			double ind1_par = ind1[param_index];
			double ind2_par = ind2[param_index];

			ind1[param_index] = ind1_par * alpha + ind2_par * (1. - alpha);
			ind2[param_index] = ind2_par * alpha + ind1_par * (1. - alpha);

			if (!static_cast<int>(limits[param_index][2])) {
				if (ind1[param_index] < limits[param_index][0])
					ind1[param_index] = limits[param_index][0];
				else if (ind1[param_index] > limits[param_index][1])
					ind1[param_index] = limits[param_index][1];

				if (ind2[param_index] < limits[param_index][0])
					ind2[param_index] = limits[param_index][0];
				else if (ind2[param_index] > limits[param_index][1])
					ind2[param_index] = limits[param_index][1];
			}
		}
	}
}

void crossover_op_arithmetic_multiple_fix::operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> distr_alpha(0, 1);

	size_t sel_params = !n || n > limits.size() ? limits.size() : n;

	for (size_t pop_index = 0; pop_index < population.size() - 1; pop_index += 2) {

		std::vector<double> &ind1 = population[pop_index];
		std::vector<double> &ind2 = population[pop_index + 1];

		std::vector<size_t> param_indexes(limits.size());
		std::iota(param_indexes.begin(), param_indexes.end(), 0);
		std::shuffle(param_indexes.begin(), param_indexes.end(), mt);
		param_indexes.resize(sel_params);

		for (const auto &param_index : param_indexes) {

			double alpha = distr_alpha(mt);

			double ind1_par = ind1[param_index];
			double ind2_par = ind2[param_index];

			ind1[param_index] = ind1_par * alpha + ind2_par * (1. - alpha);
			ind2[param_index] = ind2_par * alpha + ind1_par * (1. - alpha);

			if (!static_cast<int>(limits[param_index][2])) {
				if (ind1[param_index] < limits[param_index][0])
					ind1[param_index] = limits[param_index][0];
				else if (ind1[param_index] > limits[param_index][1])
					ind1[param_index] = limits[param_index][1];

				if (ind2[param_index] < limits[param_index][0])
					ind2[param_index] = limits[param_index][0];
				else if (ind2[param_index] > limits[param_index][1])
					ind2[param_index] = limits[param_index][1];
			}
		}
	}
}

void crossover_op_arithmetic_multiple_rnd::operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<size_t> distr_sel_params(1, !n || n > limits.size() ? limits.size() : n);
	std::uniform_real_distribution<double> distr_alpha(0, 1);

	for (size_t pop_index = 0; pop_index < population.size() - 1; pop_index += 2) {

		std::vector<double> &ind1 = population[pop_index];
		std::vector<double> &ind2 = population[pop_index + 1];

		std::vector<size_t> param_indexes(limits.size());
		std::iota(param_indexes.begin(), param_indexes.end(), 0);
		std::shuffle(param_indexes.begin(), param_indexes.end(), mt);
		param_indexes.resize(distr_sel_params(mt));

		for (const auto &param_index : param_indexes) {

			double alpha = distr_alpha(mt);

			double ind1_par = ind1[param_index];
			double ind2_par = ind2[param_index];

			ind1[param_index] = ind1_par * alpha + ind2_par * (1. - alpha);
			ind2[param_index] = ind2_par * alpha + ind1_par * (1. - alpha);

			if (!static_cast<int>(limits[param_index][2])) {
				if (ind1[param_index] < limits[param_index][0])
					ind1[param_index] = limits[param_index][0];
				else if (ind1[param_index] > limits[param_index][1])
					ind1[param_index] = limits[param_index][1];

				if (ind2[param_index] < limits[param_index][0])
					ind2[param_index] = limits[param_index][0];
				else if (ind2[param_index] > limits[param_index][1])
					ind2[param_index] = limits[param_index][1];
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// mutation operators
////////////////////////////////////////////////////////////////////////////////

void mutation_op_uniform_single::operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const
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

void mutation_op_normal_single::operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const
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
			ind[param_index] += distr_param[param_index](mt);
			if (!static_cast<int>(limits[param_index][2])) {
				if (ind[param_index] < limits[param_index][0])
					ind[param_index] = limits[param_index][0];
				else if (ind[param_index] > limits[param_index][1])
					ind[param_index] = limits[param_index][1];
			}
		}
	}
}

void mutation_op_normal_all::operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<size_t> distr_pop_index(0, population.size() / p - 1);
	std::vector<std::normal_distribution<double>> distr_param;

	for (const auto &limit : limits)
		distr_param.push_back(std::normal_distribution<double>(0., c * (limit[1] - limit[0])));

	for (auto &ind : population) {
		size_t pop_index = distr_pop_index(mt);
		if (pop_index < population.size()) {
			for (size_t param_index = 0; param_index < limits.size(); ++param_index) {
				ind[param_index] += distr_param[param_index](mt);
				if (!static_cast<int>(limits[param_index][2])) {
					if (ind[param_index] < limits[param_index][0])
						ind[param_index] = limits[param_index][0];
					else if (ind[param_index] > limits[param_index][1])
						ind[param_index] = limits[param_index][1];
				}
			}
		}
	}
}

void mutation_op_normal_multiple_fix::operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<size_t> distr_pop_index(0, population.size() / p - 1);
	std::vector<std::normal_distribution<double>> distr_param;

	size_t sel_params = !n || n > limits.size() ? limits.size() : n;

	for (const auto &limit : limits)
		distr_param.push_back(std::normal_distribution<double>(0., c * (limit[1] - limit[0])));

	for (auto &ind : population) {
		size_t pop_index = distr_pop_index(mt);
		if (pop_index < population.size()) {
			std::vector<size_t> param_indexes(limits.size());
			std::iota(param_indexes.begin(), param_indexes.end(), 0);
			std::shuffle(param_indexes.begin(), param_indexes.end(), mt);
			param_indexes.resize(sel_params);
			for (const auto &param_index : param_indexes) {
				ind[param_index] += distr_param[param_index](mt);
				if (!static_cast<int>(limits[param_index][2])) {
					if (ind[param_index] < limits[param_index][0])
						ind[param_index] = limits[param_index][0];
					else if (ind[param_index] > limits[param_index][1])
						ind[param_index] = limits[param_index][1];
				}
			}
		}
	}
}

void mutation_op_normal_multiple_rnd::operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<size_t> distr_pop_index(0, population.size() / p - 1);
	std::uniform_int_distribution<size_t> distr_sel_params(1, !n || n > limits.size() ? limits.size() : n);
	std::vector<std::normal_distribution<double>> distr_param;

	for (const auto &limit : limits)
		distr_param.push_back(std::normal_distribution<double>(0., c * (limit[1] - limit[0])));

	for (auto &ind : population) {
		size_t pop_index = distr_pop_index(mt);
		if (pop_index < population.size()) {
			std::vector<size_t> param_indexes(limits.size());
			std::iota(param_indexes.begin(), param_indexes.end(), 0);
			std::shuffle(param_indexes.begin(), param_indexes.end(), mt);
			param_indexes.resize(distr_sel_params(mt));
			for (const auto &param_index : param_indexes) {
				ind[param_index] += distr_param[param_index](mt);
				if (!static_cast<int>(limits[param_index][2])) {
					if (ind[param_index] < limits[param_index][0])
						ind[param_index] = limits[param_index][0];
					else if (ind[param_index] > limits[param_index][1])
						ind[param_index] = limits[param_index][1];
				}
			}
		}
	}
}

void mutation_op_normal_multiple_rnd2::operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_int_distribution<size_t> distr_pop_index(0, population.size() / p - 1);
	std::uniform_int_distribution<size_t> distr_param_index(0, limits.size() / q - 1);
	std::vector<std::normal_distribution<double>> distr_param;

	for (const auto &limit : limits)
		distr_param.push_back(std::normal_distribution<double>(0., c * (limit[1] - limit[0])));

	for (auto &ind : population) {
		if (distr_pop_index(mt) < population.size()) {
			for (size_t param_index = 0; param_index < limits.size(); ++param_index) {
				if (distr_param_index(mt) < limits.size()) {
					ind[param_index] += distr_param[param_index](mt);
					if (!static_cast<int>(limits[param_index][2])) {
						if (ind[param_index] < limits[param_index][0])
							ind[param_index] = limits[param_index][0];
						else if (ind[param_index] > limits[param_index][1])
							ind[param_index] = limits[param_index][1];
					}
				}
			}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////
// score scalers
////////////////////////////////////////////////////////////////////////////////

void score_scaler_none::operator()(const std::vector<double> &scores, std::vector<double> &scores_scaled) const
{
	std::copy(scores.begin(), scores.end(), scores_scaled.begin());
}

void score_scaler_offset::operator()(const std::vector<double> &scores, std::vector<double> &scores_scaled) const
{
	double min = *std::min_element(scores.begin(), scores.end());
	double max = *std::max_element(scores.begin(), scores.end());

	if (max - min == std::numeric_limits<double>::infinity())
		std::transform(scores.begin(), scores.end(), scores_scaled.begin(), [&min](const double &score){return 0.5 * score - 0.5 * min;});
	else
		std::transform(scores.begin(), scores.end(), scores_scaled.begin(), [&min](const double &score){return score - min;});
}

void score_scaler_offset_nz::operator()(const std::vector<double> &scores, std::vector<double> &scores_scaled) const
{
	double min = *std::min_element(scores.begin(), scores.end());
	double max = *std::max_element(scores.begin(), scores.end());

	if (max - min == std::numeric_limits<double>::infinity())
		std::transform(scores.begin(), scores.end(), scores_scaled.begin(), [&min](const double &score){return 0.5 * score - 0.5 * min;});
	else
		std::transform(scores.begin(), scores.end(), scores_scaled.begin(), [&min](const double &score){return score - min;});

	std::transform(scores_scaled.begin(), scores_scaled.end(), scores_scaled.begin(), [](const double &score){return score == 0.0 ? std::numeric_limits<double>::min() : score;});
}

void score_scaler_linear::operator()(const std::vector<double> &scores, std::vector<double> &scores_scaled) const
{
	double min = *std::min_element(scores.begin(), scores.end());
	double max = *std::max_element(scores.begin(), scores.end());

	if (max - min == std::numeric_limits<double>::infinity())
		std::transform(scores.begin(), scores.end(), scores_scaled.begin(), [&min](const double &score){return 0.5 * score - 0.5 * min;});
	else
		std::transform(scores.begin(), scores.end(), scores_scaled.begin(), [&min](const double &score){return score - min;});

	max = *std::max_element(scores_scaled.begin(), scores_scaled.end());

	if (max != 0.0)
		std::transform(scores_scaled.begin(), scores_scaled.end(), scores_scaled.begin(), [&max](const double &score){return score / max;});
}

void score_scaler_linear_nz::operator()(const std::vector<double> &scores, std::vector<double> &scores_scaled) const
{
	double min = *std::min_element(scores.begin(), scores.end());
	double max = *std::max_element(scores.begin(), scores.end());

	if (max - min == std::numeric_limits<double>::infinity())
		std::transform(scores.begin(), scores.end(), scores_scaled.begin(), [&min](const double &score){return 0.5 * score - 0.5 * min;});
	else
		std::transform(scores.begin(), scores.end(), scores_scaled.begin(), [&min](const double &score){return score - min;});

	max = *std::max_element(scores_scaled.begin(), scores_scaled.end());

	if (max != 0.0)
		std::transform(scores_scaled.begin(), scores_scaled.end(), scores_scaled.begin(), [&max](const double &score){return score / max;});

	std::transform(scores_scaled.begin(), scores_scaled.end(), scores_scaled.begin(), [](const double &score){return score == 0.0 ? std::numeric_limits<double>::min() : score;});
}

////////////////////////////////////////////////////////////////////////////////
// genetic algorithms
////////////////////////////////////////////////////////////////////////////////

ga_simple::ga_simple(const std::vector<std::vector<double>> &limits,
                     const selection_op &selection, const crossover_op &crossover, const mutation_op &mutation, const score_scaler &scaler,
                     const size_t &popsize, const size_t &elisize,
                     const size_t &genmax, const size_t &convn, const double &convmax, const double &scoremax,
                     const std::filesystem::path &stopfile, const size_t &thnum) :
	limits(limits),
	selection(selection),
	crossover(crossover),
	mutation(mutation),
	scaler(scaler),
	popsize(popsize % 2 ? popsize + 1: popsize),
	elisize(elisize),
	genmax(genmax),
	convn(convn),
	convmax(convmax),
	scoremax(scoremax),
	stopfile(stopfile),
	thnum(thnum > 0 ? thnum : std::thread::hardware_concurrency())
{}

// conv = bestscore[current - convn] / bestscore[current]
void ga_simple::operator()(const evaluator_single &eval, const statistics_listener &stats_listener, std::vector<double> &params, double &score,
                           std::vector<std::vector<double>> &population, std::vector<double> &scores, std::vector<size_t> &i_scores) const
{
	std::vector<std::vector<double>> elite(elisize, std::vector<double>(limits.size()));
	std::vector<double> scores_scaled(popsize);
	std::queue<double> best_scores;

	double mean_score;
	double median_score;
	double conv;

	size_t gencnt = 0;

	GANN_DBG("running simple ga with single-evaluator..." << std::endl);
	GANN_DBG("running threads: " << thnum << std::endl);

	if (population.empty()) {
		population.resize(popsize, std::vector<double>(limits.size()));
		initialize_population(population);
	}

	scores.resize(popsize);
	i_scores.resize(popsize);

	checkfinite(population, "not-finite number in population");
	checkparamsetswithinlimits(limits, population, "population parameters not within limits");

	if (thnum == 1)
		calculate_scores(eval, population, scores, scores_scaled);
	else
		calculate_scores_mt(eval, population, scores, scores_scaled);

	checkfinite(scores, "not-finite number in scores");
	checkfinite(scores_scaled, "not-finite number in scaled scores");

	++gencnt;

	calculate_stats(scores, i_scores, mean_score, median_score);
	calculate_convergence(conv, best_scores, scores[i_scores.front()]);

	//GANN_DBG("gen: " << gencnt << ", best: " << scores[i_scores.front()] << ", worst: " << scores[i_scores.back()]
	//                 << ", mean: " << mean_score << ", median: " << median_score << ", conv: " << conv << std::endl);

	if (stats_listener) {
		statistics stats{gencnt, scores[i_scores.front()], scores[i_scores.back()], mean_score, median_score, conv};
		stats_listener(stats);
	}

	while (true) {

		if (genmax > 0 && gencnt >= genmax) {
			GANN_DBG("maximum number of generations reached" << std::endl);
			break;
		}

		if (std::isnormal(conv) && conv >= convmax) {
			GANN_DBG("maximum convergence reached" << std::endl);
			break;
		}

		if (!std::isnan(scoremax) && scores[i_scores.front()] >= scoremax) {
			GANN_DBG("maximum score reached" << std::endl);
			break;
		}

		if (std::filesystem::exists(stopfile)) {
			GANN_DBG("stop file found" << std::endl);
			break;
		}

		for (size_t i = 0; i < elisize; ++i)
			elite[i] = population[i_scores[i]];

		selection(i_scores, scores_scaled, population);

		//if (size_t dups = find_2by2_duplicates(population))
		//	GANN_DBG("2by2 duplicates after selection: " << dups << std::endl);

		crossover(limits, population);

		//if (size_t dups = find_2by2_duplicates(population))
		//	GANN_DBG("2by2 duplicates after crossover: " << dups << std::endl);

		mutation(limits, population);

		//if (size_t dups = find_2by2_duplicates(population))
		//	GANN_DBG("2by2 duplicates after mutation: " << dups << std::endl);

		for (size_t i = 0; i < elisize; ++i)
			population[i] = elite[i];

		//if (size_t dups = find_2by2_duplicates(population))
		//	GANN_DBG("2by2 duplicates after elitism: " << dups << std::endl);

		checkfinite(population, "not-finite number in population");
		checkparamsetswithinlimits(limits, population, "population parameters not within limits");

		if (thnum == 1)
			calculate_scores(eval, population, scores, scores_scaled);
		else
			calculate_scores_mt(eval, population, scores, scores_scaled);

		checkfinite(scores, "not-finite number in scores");
		checkfinite(scores_scaled, "not-finite number in scaled scores");

		++gencnt;

		calculate_stats(scores, i_scores, mean_score, median_score);
		calculate_convergence(conv, best_scores, scores[i_scores.front()]);

		//GANN_DBG("gen: " << gencnt << ", best: " << scores[i_scores.front()] << ", worst: " << scores[i_scores.back()]
		//                 << ", mean: " << mean_score << ", median: " << median_score << ", conv: " << conv << std::endl);

		if (stats_listener) {
			statistics stats{gencnt, scores[i_scores.front()], scores[i_scores.back()], mean_score, median_score, conv};
			stats_listener(stats);
		}
	}

	params = population[i_scores.front()];
	score  = scores[i_scores.front()];
}

void ga_simple::operator()(const evaluator_single &eval, const statistics_listener &stats_listener, std::vector<double> &params, double &score,
                           std::vector<std::vector<double>> &population) const
{
	std::vector<double> scores;
	std::vector<size_t> i_scores;

	(*this)(eval, stats_listener, params, score, population, scores, i_scores);
}

void ga_simple::operator()(const evaluator_single &eval, const statistics_listener &stats_listener, std::vector<double> &params, double &score) const
{
	std::vector<std::vector<double>> population;
	std::vector<double> scores;
	std::vector<size_t> i_scores;

	(*this)(eval, stats_listener, params, score, population, scores, i_scores);
}

// conv = bestscore[current - convn] / bestscore[current]
void ga_simple::operator()(const evaluator_multi &eval, const statistics_listener &stats_listener, std::vector<double> &params, double &score,
                           std::vector<std::vector<double>> &population, std::vector<double> &scores, std::vector<size_t> &i_scores) const
{
	std::vector<std::vector<double>> elite(elisize, std::vector<double>(limits.size()));
	std::vector<double> scores_scaled(popsize);
	std::queue<double> best_scores;

	double mean_score;
	double median_score;
	double conv;

	size_t gencnt = 0;

	GANN_DBG("running simple ga with multi-evaluator..." << std::endl);

	if (population.empty()) {
		population.resize(popsize, std::vector<double>(limits.size()));
		initialize_population(population);
	}

	scores.resize(popsize);
	i_scores.resize(popsize);

	checkfinite(population, "not-finite number in population");
	checkparamsetswithinlimits(limits, population, "population parameters not within limits");

	eval(population, scores);
	scaler(scores, scores_scaled);

	checkfinite(scores, "not-finite number in scores");
	checkfinite(scores_scaled, "not-finite number in scaled scores");

	++gencnt;

	calculate_stats(scores, i_scores, mean_score, median_score);
	calculate_convergence(conv, best_scores, scores[i_scores.front()]);

	//GANN_DBG("gen: " << gencnt << ", best: " << scores[i_scores.front()] << ", worst: " << scores[i_scores.back()]
	//                 << ", mean: " << mean_score << ", median: " << median_score << ", conv: " << conv << std::endl);

	if (stats_listener) {
		statistics stats{gencnt, scores[i_scores.front()], scores[i_scores.back()], mean_score, median_score, conv};
		stats_listener(stats);
	}

	while (true) {

		if (genmax > 0 && gencnt >= genmax) {
			GANN_DBG("maximum number of generations reached" << std::endl);
			break;
		}

		if (std::isnormal(conv) && conv >= convmax) {
			GANN_DBG("maximum convergence reached" << std::endl);
			break;
		}

		if (!std::isnan(scoremax) && scores[i_scores.front()] >= scoremax) {
			GANN_DBG("maximum score reached" << std::endl);
			break;
		}

		if (std::filesystem::exists(stopfile)) {
			GANN_DBG("stop file found" << std::endl);
			break;
		}

		for (size_t i = 0; i < elisize; ++i)
			elite[i] = population[i_scores[i]];

		selection(i_scores, scores_scaled, population);

		//if (size_t dups = find_2by2_duplicates(population))
		//	GANN_DBG("2by2 duplicates after selection: " << dups << std::endl);

		crossover(limits, population);

		//if (size_t dups = find_2by2_duplicates(population))
		//	GANN_DBG("2by2 duplicates after crossover: " << dups << std::endl);

		mutation(limits, population);

		//if (size_t dups = find_2by2_duplicates(population))
		//	GANN_DBG("2by2 duplicates after mutation: " << dups << std::endl);

		for (size_t i = 0; i < elisize; ++i)
			population[i] = elite[i];

		//if (size_t dups = find_2by2_duplicates(population))
		//	GANN_DBG("2by2 duplicates after elitism: " << dups << std::endl);

		checkfinite(population, "not-finite number in population");
		checkparamsetswithinlimits(limits, population, "population parameters not within limits");

		eval(population, scores);
		scaler(scores, scores_scaled);

		checkfinite(scores, "not-finite number in scores");
		checkfinite(scores_scaled, "not-finite number in scaled scores");

		++gencnt;

		calculate_stats(scores, i_scores, mean_score, median_score);
		calculate_convergence(conv, best_scores, scores[i_scores.front()]);

		//GANN_DBG("gen: " << gencnt << ", best: " << scores[i_scores.front()] << ", worst: " << scores[i_scores.back()]
		//                 << ", mean: " << mean_score << ", median: " << median_score << ", conv: " << conv << std::endl);

		if (stats_listener) {
			statistics stats{gencnt, scores[i_scores.front()], scores[i_scores.back()], mean_score, median_score, conv};
			stats_listener(stats);
		}
	}

	params = population[i_scores.front()];
	score  = scores[i_scores.front()];
}

void ga_simple::operator()(const evaluator_multi &eval, const statistics_listener &stats_listener, std::vector<double> &params, double &score,
                           std::vector<std::vector<double>> &population) const
{
	std::vector<double> scores;
	std::vector<size_t> i_scores;

	(*this)(eval, stats_listener, params, score, population, scores, i_scores);
}

void ga_simple::operator()(const evaluator_multi &eval, const statistics_listener &stats_listener, std::vector<double> &params, double &score) const
{
	std::vector<std::vector<double>> population;
	std::vector<double> scores;
	std::vector<size_t> i_scores;

	(*this)(eval, stats_listener, params, score, population, scores, i_scores);
}

void ga_simple::initialize_population(std::vector<std::vector<double>> &population) const
{
	std::random_device rd;
	std::mt19937 mt(rd());
	std::vector<std::uniform_real_distribution<double>> distr;

	for (const auto &limit : limits)
		distr.push_back(std::uniform_real_distribution<double>(limit[0], limit[1]));

	for (size_t i = 0; i < population.size(); ++i)
		for (size_t j = 0; j < population[i].size(); ++j)
			population[i][j] = distr[j](mt); // population[i][j] = (limits[j][0] + limits[j][1]) / 2;
}

void ga_simple::calculate_scores(const evaluator_single &eval, const std::vector<std::vector<double>> &population, std::vector<double> &scores, std::vector<double> &scores_scaled) const
{
	for (size_t i = 0; i < population.size(); ++i) {
		double score;
		eval(population[i], score);
		scores[i] = score;
	}

	scaler(scores, scores_scaled);
}

void ga_simple::calculate_scores_mt(const evaluator_single &eval, const std::vector<std::vector<double>> &population, std::vector<double> &scores, std::vector<double> &scores_scaled) const
{
	size_t index = 0;
	std::mutex mutex;
	std::vector<std::thread> threads;
	std::vector<std::exception_ptr> eptrs(thnum);

	threads.reserve(thnum);

	for (size_t i = 0; i < thnum; ++i)
		threads.push_back(std::thread(evaluator_runner, std::cref(eval),
		                  std::ref(mutex), std::cref(population), std::ref(scores), std::ref(index),
		                  std::ref(eptrs[i])));

	for (auto &thread : threads)
		thread.join();

	std::string errmsg;
	for (size_t i = 0; i < thnum; ++i) {
		if (eptrs[i]) {
			try {
				std::rethrow_exception(eptrs[i]);
			} catch (const std::exception & e) {
				errmsg.append(" #").append(std::to_string(i)).append(": ").append(e.what());
			} catch (...) {
				errmsg.append(" #").append(std::to_string(i)).append(": <unknown exception>");
			}
		}
	}
	if (!errmsg.empty())
		throw std::runtime_error("some evaluator runners failed:" + errmsg);

	scaler(scores, scores_scaled);
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

size_t ga_simple::find_2by2_duplicates(const std::vector<std::vector<double>> &population) const
{
	size_t dups = 0;
	for (size_t i = 0; i < population.size() - 1; i += 2)
		if (population[i] == population[i + 1])
			++dups;
	return dups;
}

void ga_simple::evaluator_runner(const evaluator_single &eval, std::mutex &mutex, const std::vector<std::vector<double>> &population, std::vector<double> &scores, size_t &index, std::exception_ptr &eptr)
try {
	size_t i;
	double score;
	std::unique_lock<std::mutex> lock(mutex);

	for (;;) {

		if (index >= population.size())
			break;

		i = index++;

		lock.unlock();

		eval(population[i], score);

		lock.lock();

		scores[i] = score;
	}

} catch (...) {
	eptr = std::current_exception();
}

////////////////////////////////////////////////////////////////////////////////
// multi-layer perceptron
////////////////////////////////////////////////////////////////////////////////

mlp::mlp() : network()
{
}

mlp::mlp(const mlp &p) : network(p.network)
{
}

mlp::mlp(mlp &&p) : network(std::move(p.network))
{
}

mlp& mlp::operator=(const mlp &p)
{
	if (this != &p)
		network  = p.network;
	return *this;
}

mlp& mlp::operator=(mlp &&p)
{
	if (this != &p)
		network  = std::move(p.network);
	return *this;
}

bool mlp::operator==(const mlp &p) const
{
	return network == p.network;
}

bool mlp::operator!=(const mlp &p) const
{
	return !(*this == p);
}

void mlp::set_architecture(const std::vector<size_t> &arch)
{
	if (arch.size() < 2)
		throw std::runtime_error("too few items in mlp architecture descriptor");

	for (const auto &i : arch)
		if (i == 0)
			throw std::runtime_error("zero item in mlp architecture descriptor");

	network.resize(arch.size() - 1);

	for (size_t i = 1; i < arch.size(); ++i)
		network[i - 1] = std::vector<std::pair<std::vector<double>, activation_function>>(arch[i], std::pair<std::vector<double>,activation_function>(std::vector<double>(arch[i - 1] + 1), af_identity));
}

std::vector<size_t> mlp::get_architecture() const
{
	if (network.empty())
		return {};

	std::vector<size_t> arch(network.size() + 1);

	arch[0] = network.front().front().first.size() - 1;
	for (size_t i = 0; i < network.size(); ++i)
		arch[i + 1] = network[i].size();

	return arch;
}

void mlp::set_activation_functions(const activation_function &af)
{
	for (auto &layer : network)
		for (auto &neuron : layer)
			neuron.second = af;
}

void mlp::set_activation_functions(const std::vector<activation_function> &af)
{
	size_t i = 0;
	for (const auto &layer : network)
		i += layer.size();

	if (af.size() != i)
		throw std::runtime_error("number of activation functions doesn't match the number of neurons");

	i = 0;
	for (auto &layer : network)
		for (auto &neuron : layer)
			neuron.second = af[i++];
}

void mlp::set_activation_functions_by_layers(const std::vector<activation_function> &af)
{
	if (af.size() != network.size())
		throw std::runtime_error("number of activation functions doesn't match the number of layers");

	for (size_t i = 0; i < af.size(); ++i)
		for (auto &neuron : network[i])
			neuron.second = af[i];
}

void mlp::set_weights(const std::vector<double> &weights)
{
	size_t i = 0;
	for (const auto &layer : network)
		for (const auto &neuron : layer)
			i += neuron.first.size();

	if (weights.size() != i)
		throw std::runtime_error("number of weights doesn't match");

	i = 0;
	for (auto &layer : network)
		for (auto &neuron : layer)
			for (auto &weight : neuron.first)
				weight = weights[i++];
}

std::vector<double> mlp::get_weights() const
{
	std::vector<double> weights;

	for (const auto &layer : network)
		for (const auto &neuron : layer)
			for (const auto &weight : neuron.first)
				weights.push_back(weight);

	return weights;
}

std::vector<double> mlp::propagate(const std::vector<double> &in) const
{
	if (network.empty())
		throw std::runtime_error("empty network");

	std::vector<double> inputs = in;
	std::vector<double> outputs;

	for (const auto &layer : network) {

		outputs.clear();

		for (const auto &neuron : layer) {

			if (inputs.size() != neuron.first.size() - 1)
				throw std::runtime_error("number of inputs doesn't match the number of weights");

			double output = 0;
			for (size_t i = 0; i < neuron.first.size() - 1; ++i)
				output += neuron.first[i] * inputs[i];

			outputs.push_back(neuron.second(output + neuron.first.back()));
		}

		inputs = outputs;
	}

	return outputs;
}

double mlp::af_identity(double x)
{
	return x;
}

double mlp::af_step(double x)
{
	return x < 0. ? 0. : 1.;
}

double mlp::af_symmetric_step(double x)
{
	return x < 0. ? -1. : 1.;
}

double mlp::af_logistic(double x)
{
	return 1. / (1. + std::exp(-x));
}

double mlp::af_tanh(double x)
{
	return std::tanh(x);
}

std::ostream& operator<<(std::ostream &os, const mlp &p)
{
	for (size_t i_layer = 0; i_layer < p.network.size(); ++i_layer) {
		os << "layer" << std::endl;
		for (size_t i_neuron = 0; i_neuron < p.network[i_layer].size(); ++i_neuron) {
			os << "  neuron : " << p.network[i_layer][i_neuron].first << ", ";
			const auto &af = p.network[i_layer][i_neuron].second;
			if (af == mlp::af_identity)
				os << "identity";
			else if (af == mlp::af_step)
				os << "step";
			else if (af == mlp::af_symmetric_step)
				os << "symmetric_step";
			else if (af == mlp::af_logistic)
				os << "logistic";
			else if (af == mlp::af_tanh)
				os << "tanh";
			else
				os << "unknown";
			if (i_neuron < p.network[i_layer].size() - 1 || i_layer < p.network.size() - 1)
				os << std::endl;
		}
	}

	return os;
}

} // namespace gann

