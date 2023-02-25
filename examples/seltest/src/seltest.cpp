#include <gann/gann.hpp>

#include <map>
#include <chrono>
#include <vector>
#include <iomanip>
#include <numeric>
#include <iostream>
#include <algorithm>

std::vector<size_t> create_i_scores(const std::vector<double> &scores)
{
	std::vector<size_t> i_scores(scores.size());
	std::iota(i_scores.begin(), i_scores.end(), static_cast<size_t>(0));
	std::sort(i_scores.begin(), i_scores.end(), [&scores](const size_t &x, const size_t &y){return scores[x] > scores[y];});

	return i_scores;
}

std::vector<size_t> create_iscores(const std::vector<size_t> i_scores, const std::vector<double> &scores)
{
	std::vector<size_t> iscores(scores.size());
	for (size_t i = 0; i < scores.size(); ++i)
		iscores[i_scores[i]] = scores.size() - i;

	return iscores;
}

void test(const gann::selection_op &sel, const std::string &name,
	const size_t & runs, const std::vector<size_t> &i_scores, const std::vector<size_t> &iscores, const std::vector<double> &scores, const std::vector<std::vector<double>> &population)
{
	std::map<int, int> histogram;

	auto begin = std::chrono::steady_clock::now();
	for (int i = 0; i < runs; ++i) {
		auto pop = population;
		sel(i_scores, scores, pop);
		for (const auto & individual : pop)
			++histogram[individual[0]];
	}
	auto end = std::chrono::steady_clock::now();

	int histogram_sum = 0.;
	for (auto it = histogram.begin(); it != histogram.end(); ++it)
		histogram_sum += it->second;

	double scores_sum = 0.;
	for (const auto & score: scores)
		scores_sum += score;

	size_t iscores_sum = 0.;
	for (const auto & iscore: iscores)
		iscores_sum += iscore;

	std::cout << "selection     : " << name << std::endl;
	std::cout << "runs          : " << runs << std::endl;
	std::cout << "popsize       : " << population.size() << std::endl;
	std::cout << "duration [ms] : " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << std::endl;

#if 1
	std::cout << "ind|score|iscore|  score_p| iscore_p|   sel_p" << std::endl;
	std::cout << "---+-----+------+---------+---------+--------" << std::endl;
	for (auto it = histogram.begin(); it != histogram.end(); ++it)
		std::cout
			<< std::setw(3) << it->first
			<< "| " << std::setw(4) << scores.at(it->first)
			<< "| " << std::setw(5) << iscores.at(it->first)
			<< "| " << std::setprecision(6) << std::setw(8) << 100. * scores.at(it->first) / scores_sum
			<< "| " << std::setprecision(6) << std::setw(8) << 100. * iscores.at(it->first) / iscores_sum
			<< "| " << std::setprecision(6) << std::setw(8) << 100. * it->second / histogram_sum << std::endl;

#endif

	std::cout << std::endl;
}

int main()
{
/// Note:
/// Population must! be a vector of single element vectors, where
/// the first individual has its single element equal zero,
/// the second individual has its single element equal one,
/// the third individual has its single element equal two,
/// etc.

#if 1
	const size_t runs = 100000;

	//const std::vector<double> scores{0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	//const std::vector<double> scores{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	const std::vector<double> scores{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	//const std::vector<double> scores{10, 9, 8, 7, 6, 5, 4, 3, 2, 1};
	//const std::vector<double> scores{1, 2, 9, 9, 8, 8, 4, 3, 2, 1};

	const std::vector<std::vector<double>> population{{0},{1},{2},{3},{4},{5},{6},{7},{8},{9}};

#else
	const size_t runs = 10000;
	std::vector<double> scores;
	std::vector<std::vector<double>> population;
	for (size_t i = 0; i < 100; ++i) {
		scores.push_back(i + 1);
		population.push_back({(double)i});
	}
#endif

	const std::vector<size_t> i_scores = create_i_scores(scores);
	const std::vector<size_t> iscores = create_iscores(i_scores, scores);

	test(gann::selection_op_roulette{},   "roulette",   runs, i_scores, iscores, scores, population);
	test(gann::selection_op_rank{},       "rank",       runs, i_scores, iscores, scores, population);
	test(gann::selection_op_tournament{}, "tournament", runs, i_scores, iscores, scores, population);

	return EXIT_SUCCESS;
}
