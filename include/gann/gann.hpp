#ifndef GANN_HPP
#define GANN_HPP

#include <cmath>
#include <mutex>
#include <queue>
#include <vector>
#include <limits>
#include <ostream>
#include <exception>
#include <stdexcept>
#include <functional>

namespace gann
{
////////////////////////////////////////////////////////////////////////////////
// utils
////////////////////////////////////////////////////////////////////////////////

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
bool approximately_equal(T a, T b)
{
	return abs(a - b) <= ((abs(a) < abs(b) ? abs(b) : abs(a)) * std::numeric_limits<T>::epsilon());
}

template<typename T>
bool essentially_equal(T a, T b)
{
	return abs(a - b) <= ((abs(a) > abs(b) ? abs(b) : abs(a)) * std::numeric_limits<T>::epsilon());
}

template<typename T>
bool definitely_greater_than(T a, T b)
{
	return (a - b) > ((abs(a) < abs(b) ? abs(b) : abs(a)) * std::numeric_limits<T>::epsilon());
}

template<typename T>
bool definitely_less_than(T a, T b)
{
	return (b - a) > ((abs(a) < abs(b) ? abs(b) : abs(a)) * std::numeric_limits<T>::epsilon());
}

template<typename T>
double nz(const T &x)
{
	return x == 0.0 ? std::numeric_limits<T>::epsilon() : x;
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
		return std::numeric_limits<T>::epsilon();
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
				if (definitely_less_than(paramset[i], limits[i][0]) || definitely_greater_than(paramset[i], limits[i][1]))
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
bool checkparamsetswithinlimits(const std::vector<std::vector<T>> &limits,
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
	/// @param scores population scores
	/// @param population population genomes
	virtual void operator()(const std::vector<double> &scores, std::vector<std::vector<double>> &population) const = 0;
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
	explicit selection_op_roulette(size_t extra_runs = 1) : extra_runs(extra_runs) {};

	virtual void operator()(const std::vector<double> &scores, std::vector<std::vector<double>> &population) const override;
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

/// @brief Single crossover operator.
///
///        Each two neighbouring individuals are crossovered into new two ones.
///        Single parameter index 'i' is randomly selected.
///        Random value 'alpha' is selected from uniform distribution U(0,1).
///        Then crossover is performed in this way:
///        child1 = parent1
///        child1[i] = parent1[i] * alpha + parent2[i] * (1 - alpha)
///        child2 = parent2
///        child2[i] = parent2[i] * alpha + parent1[i] * (1 - alpha)
class crossover_op_single_arithmetic : public crossover_op
{
public:
	virtual void operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const override;
};

/// @brief Multiple crossover operator.
///
///        Each two neighbouring individuals are crossovered into new two ones.
///        Multiple parameter indexes 'i' are randomly selected.
///        For each 'i' random value 'alpha' is selected from uniform distribution U(0,1).
///        Then crossover is performed in this way:
///        child1 = parent1
///        child1[i] = parent1[i] * alpha + parent2[i] * (1 - alpha)
///        child2 = parent2
///        child2[i] = parent2[i] * alpha + parent1[i] * (1 - alpha)
class crossover_op_multiple_arithmetic : public crossover_op
{
public:
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

/// @brief Uniform mutation operator.
///
///        Each individual is selected for mutation with probability 'p'.
///        Then random parameter is selected and mutated by replacing with value
///        from uniform distribution U(limit[i][0], limit[i][1])
class mutation_op_uniform : public mutation_op
{
private:
	const double p;
public:
	/// @brief Constructor.
	/// @param p probability of mutation of one individual
	explicit mutation_op_uniform(double p = 0.01) : p(p) {};

	virtual void operator()(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const override;
};

/// @brief Normal mutation operator.
///
///        Each individual is selected for mutation with probability 'p'.
///        Then random parameter is selected and mutated by adding a value
///        from normal distribution N(0, c * (limit[i][1] - limit[i][0]))
class mutation_op_normal : public mutation_op
{
private:
	const double p;
	const double c;
public:
	/// @brief Constructor.
	/// @param p probability of mutation of one individual
	/// @param c constant used to derive standard deviation parameter of normal distribution
	///          stddev = c * (limit[i][1] - limit[i][0])
	explicit mutation_op_normal(double p = 0.01, double c = 0.25) : p(p), c(c) {};

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
	/// @param thnum number of running threads, if zero then the number will be determined automatically.
	///              This parameter is meaningful only for running with single individual evaluator.
	///              Multiple individuals evaluator runs only in one (caller's) thread.
	ga_simple(const std::vector<std::vector<double>> &limits,
	          const selection_op &selection, const crossover_op &crossover, const mutation_op &mutation, const score_scaler &scaler,
	          const size_t &popsize, const size_t &elisize,
	          const size_t &genmax, const size_t &convn, const double &convmax, const double &scoremax,
	          const size_t &thnum);

	/// @brief Runs genetic algorithm.
	/// @param eval evaluator. If thnum or number of automatically detected threads equals one,
	///             then it is executed sequentially in caller's thread,
	///             otherwise it is executed concurrently in multiple separated threads.
	/// @param stats_listener Called after each generation has been evaluated.
	/// @param params best genome
	/// @param score best score
	void operator()(const evaluator_single &eval, const statistics_listener &stats_listener,
	                std::vector<double> &params, double &score) const;

	/// @brief Runs genetic algorithm.
	/// @param eval evaluator. It is executed sequentially in caller's thread;
	/// @param stats_listener Called after each generation has been evaluated.
	/// @param params best genome
	/// @param score best score
	void operator()(const evaluator_multi &eval, const statistics_listener &stats_listener,
	                std::vector<double> &params, double &score) const;

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

