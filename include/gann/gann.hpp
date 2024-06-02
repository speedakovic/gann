#ifndef GANN_HPP
#define GANN_HPP

#include <cmath>
#include <mutex>
#include <queue>
#include <vector>
#include <limits>
#include <istream>
#include <ostream>
#include <sstream>
#include <exception>
#include <stdexcept>
#include <functional>
#include <filesystem>

namespace gann
{
////////////////////////////////////////////////////////////////////////////////
// utils
////////////////////////////////////////////////////////////////////////////////

namespace serialize
{

template<typename T>
std::ostream& operator<<(std::ostream &out, const std::vector<T> &x)
{
	for (size_t i = 0; i < x.size(); ++i) {
		out << x[i];
		if (i < x.size() - 1)
			out << " ";
	}
	return out;
}

template<typename T>
std::istream& operator>>(std::istream &in, std::vector<T> &x)
{
	x.resize(0);
	std::string line;
	if (std::getline(in, line)) {
		std::stringstream ss{line};
		T d;
		while (ss >> d)
			x.push_back(d);
	}
	return in;
}

template<typename T>
std::ostream& operator<<(std::ostream &out, const std::vector<std::vector<T>> &x)
{
	for (size_t i = 0; i < x.size(); ++i) {
		for (size_t j = 0; j < x[i].size(); ++j) {
			out << x[i][j];
			if (j < x[i].size() - 1)
				out << " ";
		}
		if (i < x.size() - 1)
			out << std::endl;
	}
	return out;
}

template<typename T>
std::istream& operator>>(std::istream &in, std::vector<std::vector<T>> &x)
{
	x.resize(0);
	std::string line;
	while (std::getline(in, line)) {
		std::vector<double> y;
		std::stringstream ss{line};
		T d;
		while (ss >> d)
			y.push_back(d);
		x.push_back(y);
	}
	return in;
}

} // namespace serialize

template<typename T>
std::ostream& operator<<(std::ostream &out, const std::vector<T> &x)
{
	out << '[';
	for (size_t i = 0; i < x.size(); ++i)
		out << x[i] << (i < x.size() - 1 ? ", " : "");
	out << "]";
	return out;
}

template<typename T>
double nz(const T &x)
{
	return x == 0.0 ? std::numeric_limits<T>::min() : x;
}

template<typename T>
double clamp(const T &x)
{
	if (x == std::numeric_limits<T>::infinity())
		return std::numeric_limits<T>::max();
	else if (x == -std::numeric_limits<T>::infinity())
		return std::numeric_limits<T>::lowest();
	else
		return x;
}

template<typename T>
double clamp_nz(const T &x)
{
	if (x == std::numeric_limits<T>::infinity())
		return std::numeric_limits<T>::max();
	else if (x == -std::numeric_limits<T>::infinity())
		return std::numeric_limits<T>::lowest();
	else if (x == 0.0)
		return std::numeric_limits<T>::min();
	else
		return x;
}


template<typename T>
bool isfinite(const std::vector<T> &v)
{
	for (const auto &x : v)
		if (!std::isfinite(x))
			return false;
	return true;
}

template<typename T>
bool isfinite(const std::vector<std::vector<T>> &vv)
{
	for (const auto &v : vv)
		for (const auto &x : v)
			if (!std::isfinite(x))
				return false;
	return true;
}

template<typename T>
bool isnormal(const std::vector<T> &v)
{
	for (const auto &x : v)
		if (!std::isnormal(x))
			return false;
	return true;
}

template<typename T>
bool isnormal(const std::vector<std::vector<T>> &vv)
{
	for (const auto &v : vv)
		for (const auto &x : v)
			if (!std::isnormal(x))
				return false;
	return true;
}

template <typename T>
bool areparamsetswithinlimits(const std::vector<std::vector<T>> &limits,
                              const std::vector<std::vector<T>> &paramsets)
{
	for (size_t i = 0; i < limits.size(); ++i)
		if (!limits[i][2])
			for (const auto &paramset : paramsets)
				if (std::isless(paramset[i], limits[i][0]) || std::isgreater(paramset[i], limits[i][1]))
					return false;
	return true;
}

template<typename T>
void checkfinite(const T &x, const std::string &msg = "not-finite number")
{
	if (!std::isfinite(x))
		throw std::runtime_error(msg);
}

template<typename T>
void checkfinite(const std::vector<T> &v, const std::string &msg = "not-finite number")
{
	if (!isfinite(v))
		throw std::runtime_error(msg);
}

template<typename T>
void checkfinite(const std::vector<std::vector<T>> &vv, const std::string &msg = "not-finite number")
{
	if (!isfinite(vv))
		throw std::runtime_error(msg);
}

template<typename T>
void checknormal(const T &x, const std::string &msg = "not-normal number")
{
	if (!std::isnormal(x))
		throw std::runtime_error(msg);
}

template<typename T>
void checknormal(const std::vector<T> &v, const std::string &msg = "not-normal number")
{
	if (!isnormal(v))
		throw std::runtime_error(msg);
}

template<typename T>
void checknormal(const std::vector<std::vector<T>> &vv, const std::string &msg = "not-normal number")
{
	if (!isnormal(vv))
		throw std::runtime_error(msg);
}

template<typename T>
void checkgreater(const T &x, const T &y, const std::string &msg = "not-greater number")
{
	if (!std::isgreater(x, y))
		throw std::runtime_error(msg);
}

template<typename T>
void checknotgreater(const T &x, const T &y, const std::string &msg = "greater number")
{
	if (std::isgreater(x, y))
		throw std::runtime_error(msg);
}

template<typename T>
void checkless(const T &x, const T &y, const std::string &msg = "not-less number")
{
	if (!std::isless(x, y))
		throw std::runtime_error(msg);
}

template<typename T>
void checknotless(const T &x, const T &y, const std::string &msg = "less number")
{
	if (std::isless(x, y))
		throw std::runtime_error(msg);
}

template <typename T>
void checkparamsetswithinlimits(const std::vector<std::vector<T>> &limits,
                                const std::vector<std::vector<T>> &paramsets,
                                const std::string &msg = "paramsets not within limits")
{
	if (!areparamsetswithinlimits(limits, paramsets))
		throw std::runtime_error(msg);
}

////////////////////////////////////////////////////////////////////////////////
// selection operators
////////////////////////////////////////////////////////////////////////////////

/// @brief Base class for selection operator.
///
///        The result of selection (population vector), which immediately
///        goes to crossover, should maintain the same size and be produced
///        in this form: {parent1, parent2, parent1, parent2, parent1, parent2, ...}
///        It is the selection operator's responsibility, that each parent1 shouldn't equal parent2.
class selection_op
{
public:
	/// @brief Destructor.
	virtual ~selection_op() = default;

	/// @brief Runs operator.
	/// @param i_scores sorted score indexes
	/// @param scores population scores
	/// @param population population genomes
	virtual void operator()(const std::vector<size_t> &i_scores, const std::vector<double> &scores, std::vector<std::vector<double>> &population) const = 0;
};

/// @brief Custom selection operator.
class selection_op_custom : public selection_op
{
public:
	typedef std::function<void(const std::vector<size_t> &i_scores, const std::vector<double> &scores, std::vector<std::vector<double>> &population)> func_type;
private:
	const func_type func;
public:
	/// @brief Constructor.
	/// @param func operator function
	explicit selection_op_custom(const func_type &func) : func(func) {}

	virtual void operator()(const std::vector<size_t> &i_scores, const std::vector<double> &scores, std::vector<std::vector<double>> &population) const override
	{
		func(i_scores, scores, population);
	}
};

/// @brief Roulette selection operator.
///
///        The given scores must be positive, so appropriate scaler should be used.
class selection_op_roulette : public selection_op
{
private:
	const size_t extra_runs;
public:
	/// @brief Constructor.
	/// @param extra_runs number of extra roulette wheel runs for the each
	///        second parent to be different from the first one.
	explicit selection_op_roulette(size_t extra_runs = 1) : extra_runs(extra_runs) {}

	virtual void operator()(const std::vector<size_t> &i_scores, const std::vector<double> &scores, std::vector<std::vector<double>> &population) const override;
};

/// @brief Rank selection operator.
class selection_op_rank : public selection_op
{
private:
	const size_t extra_runs;
public:
	/// @brief Constructor.
	/// @param extra_runs number of extra rank runs for the each
	///        second parent to be different from the first one.
	explicit selection_op_rank(size_t extra_runs = 1) : extra_runs(extra_runs) {}

	virtual void operator()(const std::vector<size_t> &i_scores, const std::vector<double> &scores, std::vector<std::vector<double>> &population) const override;
};

/// @brief Tournament selection operator.
class selection_op_tournament : public selection_op
{
private:
	const size_t competitors_num;
	const size_t extra_runs;
public:
	/// @brief Constructor.
	/// @param competitors_num number of competitors in each tournament
	/// @param extra_runs number of extra tournament runs for the each
	///        second parent to be different from the first one.
	explicit selection_op_tournament(size_t competitors_num = 2, size_t extra_runs = 1) : competitors_num(competitors_num), extra_runs(extra_runs) {}

	virtual void operator()(const std::vector<size_t> &i_scores, const std::vector<double> &scores, std::vector<std::vector<double>> &population) const override;
};

////////////////////////////////////////////////////////////////////////////////
// crossover operators
////////////////////////////////////////////////////////////////////////////////

/// @brief Base class for crossover operator.
///
///        The crossover operator should await the selected population
///        in this form: {parent1, parent2, parent1, parent2, parent1, parent2, ...}
///        The result of crossover (population vector), which immediately
///        goes to mutation, should maintain the same size.
class crossover_op
{
public:
	/// @brief Destructor.
	virtual ~crossover_op() = default;

	/// @brief Runs operator.
	/// @param limits genome limits
	/// @param population population genomes
	virtual void operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const = 0;
};

/// @brief Custom crossover operator.
class crossover_op_custom : public crossover_op
{
public:
	typedef std::function<void(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population)> func_type;
private:
	const func_type func;
public:
	/// @brief Constructor.
	/// @param func operator function
	explicit crossover_op_custom(const func_type &func) : func(func) {}

	virtual void operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const override
	{
		func(limits, population);
	};
};

/// @brief Arithmetic single-parameter crossover operator.
///
///        Each two neighbouring individuals are selected to crossover into new two ones.
///        Then single parameter index 'i' is randomly selected.
///        Then random value 'alpha' is selected from uniform distribution U(0,1).
///        Then crossover is performed in this way:
///        child1 = parent1
///        child1[i] = parent1[i] * alpha + parent2[i] * (1 - alpha)
///        child2 = parent2
///        child2[i] = parent2[i] * alpha + parent1[i] * (1 - alpha)
class crossover_op_arithmetic_single : public crossover_op
{
public:
	virtual void operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const override;
};

/// @brief Arithmetic all-parameters crossover operator.
///
///        Each two neighbouring individuals are selected to crossover into new two ones.
///        Then all parameter indexes 'i' are selected.
///        Then for each 'i' random value 'alpha' is selected from uniform distribution U(0,1).
///        Then crossover is performed in this way:
///        child1 = parent1
///        child1[i] = parent1[i] * alpha + parent2[i] * (1 - alpha)
///        child2 = parent2
///        child2[i] = parent2[i] * alpha + parent1[i] * (1 - alpha)
class crossover_op_arithmetic_all : public crossover_op
{
public:
	virtual void operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const override;
};

/// @brief Arithmetic multiple-parameter crossover operator, fixed number of crossovered parameters.
///
///        Each two neighbouring individuals are selected to crossover into new two ones.
///        Then 'n' number of parameter indexes 'i' are randomly selected.
///        Then for each 'i' random value 'alpha' is selected from uniform distribution U(0,1).
///        Then crossover is performed in this way:
///        child1 = parent1
///        child1[i] = parent1[i] * alpha + parent2[i] * (1 - alpha)
///        child2 = parent2
///        child2[i] = parent2[i] * alpha + parent1[i] * (1 - alpha)
class crossover_op_arithmetic_multiple_fix : public crossover_op
{
private:
	const size_t n;
public:
	/// @brief Constructor.
	/// @param n number of randomly selected parameters to be crossovered,
	///          zero will be replaced by number of all parameters
	explicit crossover_op_arithmetic_multiple_fix(size_t n = 0) : n(n) {}

	virtual void operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const override;
};

/// @brief Arithmetic multiple-parameter crossover operator, random number of crossovered parameters.
///
///        Each two neighbouring individuals are selected to crossover into new two ones.
///        Then number (choosen from uniform distribution U(1, n)) of parameter indexes 'i' are randomly selected.
///        Then for each 'i' random value 'alpha' is selected from uniform distribution U(0,1).
///        Then crossover is performed in this way:
///        child1 = parent1
///        child1[i] = parent1[i] * alpha + parent2[i] * (1 - alpha)
///        child2 = parent2
///        child2[i] = parent2[i] * alpha + parent1[i] * (1 - alpha)
class crossover_op_arithmetic_multiple_rnd : public crossover_op
{
private:
	const size_t n;
public:
	/// @brief Constructor.
	/// @param n determines number (choosen from uniform distribution U(1, n)) of randomly selected parameters to be crossovered,
	///          zero will be replaced by number of all parameters
	explicit crossover_op_arithmetic_multiple_rnd(size_t n = 0) : n(n) {}

	virtual void operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const override;
};

////////////////////////////////////////////////////////////////////////////////
// mutation operators
////////////////////////////////////////////////////////////////////////////////

/// @brief Base class for mutation operator.
///
///        The result of mutation (population vector), which immediately
///        goes to conditional elitism process and then to evaluation, should maintain the same size.
class mutation_op
{
public:
	/// @brief Destructor.
	virtual ~mutation_op() = default;

	/// @brief Runs operator.
	/// @param limits genome limits
	/// @param population population genomes
	virtual void operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const = 0;
};

/// @brief Custom mutation operator.
class mutation_op_custom : public mutation_op
{
public:
	typedef std::function<void(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population)> func_type;
private:
	const func_type func;
public:
	/// @brief Constructor.
	/// @param func operator function
	explicit mutation_op_custom(const func_type &func) : func(func) {}

	virtual void operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const override
	{
		func(limits, population);
	}
};

/// @brief Uniform single-parameter mutation operator.
///
///        Each individual is selected for mutation with probability 'p'.
///        Then single random parameter is selected and mutated by replacing with value
///        from uniform distribution U(limit[i][0], limit[i][1])
class mutation_op_uniform_single : public mutation_op
{
private:
	const double p;
public:
	/// @brief Constructor.
	/// @param p probability of mutation of one individual
	explicit mutation_op_uniform_single(double p = 0.01) : p(p) {}

	virtual void operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const override;
};

/// @brief Normal single-parameter mutation operator.
///
///        Each individual is selected for mutation with probability 'p'.
///        Then single random parameter is selected and mutated by adding value
///        from normal distribution N(0, c * (limit[i][1] - limit[i][0]))
class mutation_op_normal_single : public mutation_op
{
private:
	const double p;
	const double c;
public:
	/// @brief Constructor.
	/// @param p probability of mutation of each individual
	/// @param c constant used to derive standard deviation parameter of normal distribution
	///          stddev = c * (limit[i][1] - limit[i][0])
	explicit mutation_op_normal_single(double p = 0.01, double c = 0.25) : p(p), c(c) {}

	virtual void operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const override;
};

/// @brief Normal all-parameters mutation operator.
///
///        Each individual is selected for mutation with probability 'p'.
///        Then all parameters are mutated by adding values
///        from normal distribution N(0, c * (limit[i][1] - limit[i][0]))
class mutation_op_normal_all : public mutation_op
{
private:
	const double p;
	const double c;
public:
	/// @brief Constructor.
	/// @param p probability of mutation of each individual
	/// @param c constant used to derive standard deviation parameter of normal distribution
	///          stddev = c * (limit[i][1] - limit[i][0])
	explicit mutation_op_normal_all(double p = 0.01, double c = 0.25) : p(p), c(c) {}

	virtual void operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const override;
};

/// @brief Normal multiple-parameters mutation operator, fixed number of mutated parameters.
///
///        Each individual is selected for mutation with probability 'p'.
///        Then 'n' parameters are randomly selected and mutated by adding values
///        from normal distribution N(0, c * (limit[i][1] - limit[i][0]))
class mutation_op_normal_multiple_fix : public mutation_op
{
private:
	const double p;
	const size_t n;
	const double c;
public:
	/// @brief Constructor.
	/// @param p probability of mutation of each individual
	/// @param n number of randomly selected parameters to be mutated,
	///          zero will be replaced by number of all parameters
	/// @param c constant used to derive standard deviation parameter of normal distribution
	///          stddev = c * (limit[i][1] - limit[i][0])
	explicit mutation_op_normal_multiple_fix(double p = 0.01, size_t n = 0, double c = 0.25) : p(p), n(n), c(c) {}

	virtual void operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const override;
};

/// @brief Normal multiple-parameters mutation operator, random number of mutated parameters.
///
///        Each individual is selected for mutation with probability 'p'.
///        Then number (choosen from uniform distribution U(1, n)) of parameters
///        are randomly selected and mutated by adding values.
///        from normal distribution N(0, c * (limit[i][1] - limit[i][0]))
class mutation_op_normal_multiple_rnd : public mutation_op
{
private:
	const double p;
	const size_t n;
	const double c;
public:
	/// @brief Constructor.
	/// @param p probability of mutation of each individual
	/// @param n determines number (choosen from U(1,n)) of randomly selected parameters to be mutated,
	///          zero will be replaced by number of all parameters
	/// @param c constant used to derive standard deviation parameter of normal distribution
	///          stddev = c * (limit[i][1] - limit[i][0])
	explicit mutation_op_normal_multiple_rnd(double p = 0.01, size_t n = 0, double c = 0.25) : p(p), n(n), c(c) {}

	virtual void operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const override;
};

/// @brief Normal multiple-parameters mutation operator, random number of mutated parameters.
///
///        Each individual is selected for mutation with probability 'p'.
///        Then each parameter is selected with probability 'q' and mutated
///        by adding value from normal distribution N(0, c * (limit[i][1] - limit[i][0]))
class mutation_op_normal_multiple_rnd2 : public mutation_op
{
private:
	const double p;
	const double q;
	const double c;
public:
	/// @brief Constructor.
	/// @param p probability of mutation of each individual
	/// @param q probability of mutation of each parameter
	/// @param c constant used to derive standard deviation parameter of normal distribution
	///          stddev = c * (limit[i][1] - limit[i][0])
	explicit mutation_op_normal_multiple_rnd2(double p = 0.01, double q = 0.01, double c = 0.25) : p(p), q(q), c(c) {}

	virtual void operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const override;
};

////////////////////////////////////////////////////////////////////////////////
// score scalers
////////////////////////////////////////////////////////////////////////////////

/// @brief Base class for score scaler.
class score_scaler
{
public:
	/// @brief Destructor.
	virtual ~score_scaler() = default;

	/// @brief Runs scaler.
	/// @param scores population scores
	/// @param scores_scaled scaled population scores
	virtual void operator()(const std::vector<double> &scores, std::vector<double> &scores_scaled) const = 0;
};

/// @brief Custom scaler.
class score_scaler_custom : public score_scaler
{
public:
	typedef std::function<void(const std::vector<double> &scores, std::vector<double> &scores_scaled)> func_type;
private:
	const func_type func;
public:
	/// @brief Constructor.
	/// @param func scaler function
	explicit score_scaler_custom(const func_type &func) : func(func) {}

	virtual void operator()(const std::vector<double> &scores, std::vector<double> &scores_scaled) const override
	{
		func(scores, scores_scaled);
	}
};

/// @brief None scaler.
///        This scaler performs no scaling at all.
class score_scaler_none : public score_scaler
{
public:
	virtual void operator()(const std::vector<double> &scores, std::vector<double> &scores_scaled) const override;
};

/// @brief Offset scaler.
///        This scaler just shifts the scores minimum to zero.
class score_scaler_offset : public score_scaler
{
public:
	virtual void operator()(const std::vector<double> &scores, std::vector<double> &scores_scaled) const override;
};

/// @brief Offset scaler (non-zero).
///        This scaler just shifts the scores minimum to zero + eps.
class score_scaler_offset_nz : public score_scaler
{
public:
	virtual void operator()(const std::vector<double> &scores, std::vector<double> &scores_scaled) const override;
};

/// @brief Linear scaler.
///        This scaler scales the scores to the interval [0,1]
class score_scaler_linear : public score_scaler
{
public:
	virtual void operator()(const std::vector<double> &scores, std::vector<double> &scores_scaled) const override;
};

/// @brief Linear scaler (non-zero).
///        This scaler scales the scores to the interval (0,1]
class score_scaler_linear_nz : public score_scaler
{
public:
	virtual void operator()(const std::vector<double> &scores, std::vector<double> &scores_scaled) const override;
};

////////////////////////////////////////////////////////////////////////////////
// genetic algorithms
////////////////////////////////////////////////////////////////////////////////

/// @brief Simple genetic algorithm.
class ga_simple
{
public:
	/// @brief Statistics.
	struct statistics
	{
		size_t generation;   ///< Generation number
		double best_score;   ///< Best score
		double worst_score;  ///< Worst score
		double mean_score;   ///< Mean score
		double median_score; ///< Median score
		double convergence;  ///< Convergence
	};

	/// @brief Single individual evaluator.
	///        It computes score (fitness) for given individual genome.
	/// @param params individual genome
	/// @param score evaluated score of given genome
	typedef std::function<void(const std::vector<double> &params, double &score)> evaluator_single;

	/// @brief Multiple individuals evaluator.
	///        It computes scores (fitnesses) for given population genomes.
	/// @param params vector of the whole population genomes
	/// @param scores evaluated scores of given genomes. Vector passed to evaluator has
	///               the same size as vector of population genomes, so no insertion is needed.
	typedef std::function<void(const std::vector<std::vector<double>> &params, std::vector<double> &scores)> evaluator_multi;

	/// @brief Statistics listener.
	/// @param stats statistics
	typedef std::function<void(const statistics &stats)> statistics_listener;

private:
	const std::vector<std::vector<double>> limits;

	const selection_op &selection;
	const crossover_op &crossover;
	const mutation_op  &mutation;
	const score_scaler &scaler;

	const size_t popsize;
	const size_t elisize;

	const size_t genmax;
	const size_t convn;
	const double convmax;
	const double scoremax;
	const std::filesystem::path stopfile;

	const size_t thnum;

public:
	/// @brief Constructor.
	/// @param limits limiting values for each optimized parameter
	///               [i][0] - minimum
	///               [i][1] - maximum
	///               [i][2] - enable minimum/maximum crossing if nonzero
	/// @param selection selection operator
	/// @param crossover crossover operator
	/// @param mutation mutation operator
	/// @param scaler score scaler (the scaled scores are used by selection operator)
	/// @param popsize population size (must be even number, if it is not, then +1 will be added)
	/// @param elisize number of best individuals, that will be directly passed to next generation (elitism feature)
	/// @param genmax maximum generations, running will be terminated when this value is reached.
	///               Set to zero to disable termination based on number of generations.
	/// @param convn number of generations to look back when convergence is calculated.
	///              Set to zero to disable termination based on convergence.
	///              To use termination criterium based on convergence, scores must be positive numbers.
	/// @param convmax maximum convergence, running will be terminated when this value is reached
	/// @param scoremax maximum score, running will be terminated when this value is reached
	///                 Set to nan to disable termination based on score.
	/// @param stopfile stop file, running will be terminated when this file is found.
	///                 Set to empty path to disable termination based on stop file.
	/// @param thnum number of running threads, if zero then the number will be determined automatically.
	///              This parameter is meaningful only for running with single individual evaluator.
	///              Multiple individuals evaluator runs only in one (caller's) thread.
	ga_simple(const std::vector<std::vector<double>> &limits,
	          const selection_op &selection, const crossover_op &crossover, const mutation_op &mutation, const score_scaler &scaler,
	          const size_t &popsize, const size_t &elisize,
	          const size_t &genmax, const size_t &convn, const double &convmax, const double &scoremax,
	          const std::filesystem::path &stopfile, const size_t &thnum);

	/// @brief Runs genetic algorithm.
	/// @param eval evaluator. If thnum or number of automatically detected threads equals one,
	///             then it is executed sequentially in caller's thread,
	///             otherwise it is executed concurrently in multiple separated threads.
	/// @param stats_listener Called after each generation has been evaluated.
	/// @param params best genome/individual
	/// @param score best score
	/// @param population population to work with. If empty population is passed, then it will be initialized. Otherwise it is considered
	///                   as initialized, e.g. as result from previous run.
	/// @param scores returned scores of population
	/// @param i_scores returned indexes to scores. Indexes are sorted in way to point to scores/population from highest to lowest value of score.
	///                 i_score[0] - index of individual (in population vector) with highest score
	///                            - index of score (in scores vector) with highest value
	///                 population[i_score[0]] - individual with highest score (= params)
	///                 scores[i_score[0]] - highest score (= score)
	void operator()(const evaluator_single &eval, const statistics_listener &stats_listener, std::vector<double> &params, double &score,
	                std::vector<std::vector<double>> &population, std::vector<double> &scores, std::vector<size_t> &i_scores) const;

	void operator()(const evaluator_single &eval, const statistics_listener &stats_listener, std::vector<double> &params, double &score,
	                std::vector<std::vector<double>> &population) const;

	void operator()(const evaluator_single &eval, const statistics_listener &stats_listener, std::vector<double> &params, double &score) const;

	/// @brief Runs genetic algorithm.
	/// @param eval evaluator. It is executed sequentially in caller's thread;
	/// @param stats_listener Called after each generation has been evaluated.
	/// @param params best genome/individual
	/// @param score best score
	/// @param population population to work with. If empty population is passed, then it will be initialized. Otherwise it is considered
	///                   as initialized, e.g. as result from previous run.
	/// @param scores returned scores of population
	/// @param i_scores returned indexes to scores. Indexes are sorted in way to point to scores/population from highest to lowest value of score.
	///                 i_score[0] - index of individual (in population vector) with highest score
	///                            - index of score (in scores vector) with highest value
	///                 population[i_score[0]] - individual with highest score (= params)
	///                 scores[i_score[0]] - highest score (= score)
	void operator()(const evaluator_multi &eval, const statistics_listener &stats_listener, std::vector<double> &params, double &score,
	                std::vector<std::vector<double>> &population, std::vector<double> &scores, std::vector<size_t> &i_scores) const;

	void operator()(const evaluator_multi &eval, const statistics_listener &stats_listener, std::vector<double> &params, double &score,
	                std::vector<std::vector<double>> &population) const;

	void operator()(const evaluator_multi &eval, const statistics_listener &stats_listener, std::vector<double> &params, double &score) const;

private:
	void initialize_population(std::vector<std::vector<double>> &population) const;
	void calculate_scores(const evaluator_single &eval, const std::vector<std::vector<double>> &population, std::vector<double> &scores, std::vector<double> &scores_scaled) const;
	void calculate_scores_mt(const evaluator_single &eval, const std::vector<std::vector<double>> &population, std::vector<double> &scores, std::vector<double> &scores_scaled) const;
	void calculate_stats(const std::vector<double> &scores, std::vector<size_t> &i_scores, double &mean_score, double &median_score) const;
	void calculate_convergence(double &conv, std::queue<double> &best_scores, const double &best_score) const;
	size_t find_2by2_duplicates(const std::vector<std::vector<double>> &population) const;

	static void evaluator_runner(const evaluator_single &eval, std::mutex &mutex, const std::vector<std::vector<double>> &population, std::vector<double> &scores, size_t &index, std::exception_ptr &eptr);
};

////////////////////////////////////////////////////////////////////////////////
// multi-layer perceptron
////////////////////////////////////////////////////////////////////////////////

/// @brief Multi-layer perceptron.
class mlp
{
public:
	/// @brief Activation function.
	typedef double (*activation_function)(double);

private:
	std::vector<std::vector<std::pair<std::vector<double>, activation_function>>> network;

public:
	/// @brief Default constructor.
	mlp();

	/// @brief Copy constructor.
	mlp(const mlp &p);

	/// @brief Move constructor.
	mlp(mlp &&p);

	//!@{
	mlp& operator=(const mlp &p);
	mlp& operator=(mlp &&p);

	bool operator==(const mlp &p) const;
	bool operator!=(const mlp &p) const;
	//!@}

	/// @brief Sets network architecture.
	/// @param arch architecture descriptor
	///             arch[0] - number of inputs (length of input vector)
	///             arch[1] - number of neurons in first layer
	///             arch[2] - number of neurons in second layer
	///             arch[n] - number of neurons in nth layer
	///
	///             All neurons in layer 'i' have number of weights (inputs) equal number of neurons
	///             in layer 'i-1' plus one (last one) represeting neuron's bias.
	///
	void set_architecture(const std::vector<size_t> &arch);

	/// @brief Gets network architecture.
	/// @return architecture descriptor
	std::vector<size_t> get_architecture() const;

	/// @brief Sets activation functions.
	/// @param af one activation function to be set to all neurons
	void set_activation_functions(const activation_function &af);

	/// @brief Sets activation functions.
	/// @param af activation functions to be set,
	///           number of activation functions must be equal number of neurons
	void set_activation_functions(const std::vector<activation_function> &af);

	/// @brief Sets activation functions.
	/// @param af activation functions to be set,
	///           number of activation functions must be equal number of layers
	void set_activation_functions_by_layers(const std::vector<activation_function> &af);

	/// @brief Sets weights.
	/// @param weights weights to be set,
	///                number of weights must be equal number of weights across all existing neurons
	void set_weights(const std::vector<double> &weights);

	/// @brief Gets weights.
	/// @return weights
	std::vector<double> get_weights() const;

	/// @brief Propagates input vector through network and returns resulting output vector.
	/// @param in input vector, its size must be equal number of inputs specified in network architecture
	/// @return output vector, its size equals number of neurons in last layer
	std::vector<double> propagate(const std::vector<double> &in) const;

	/// @name Activation functions.
	//!@{
	static double af_identity(double x);
	static double af_step(double x);
	static double af_symmetric_step(double x);
	static double af_logistic(double x);
	static double af_tanh(double x);
	//!@}

	/// @brief Output stream operator.
	friend std::ostream& operator<<(std::ostream &os, const mlp &p);
};

} // namespace gann

#endif // GANN_HPP

