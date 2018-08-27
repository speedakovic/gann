#ifndef GA_H
#define GA_H

#include <queue>
#include <vector>

namespace gann
{

////////////////////////////////////////////////////////////////////////////////
// selection operators
////////////////////////////////////////////////////////////////////////////////

/// @brief Base class for selection operator.
class selection_op
{
public:
	/// @brief Destructor.
	virtual ~selection_op() {}

	/// @brief Runs operator.
	/// @param scores population scores
	/// @param population population genomes
	virtual void run(const std::vector<double> &scores, std::vector<std::vector<double>> &population) const = 0;
};

/// @brief Roulette selection operator.
///        This operator suppose the scores to be positive.
class selection_op_roulette : public selection_op
{
public:
	/// @copydoc selection_op::run
	virtual void run(const std::vector<double> &scores, std::vector<std::vector<double>> &population) const;
};

////////////////////////////////////////////////////////////////////////////////
// crossover operators
////////////////////////////////////////////////////////////////////////////////

/// @brief Base class for crossover operator.
class crossover_op
{
public:
	/// @brief Destructor.
	virtual ~crossover_op() {}

	/// @brief Runs operator.
	/// @param limits genome limits
	/// @param population population genomes
	virtual void run(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const = 0;
};

/// @brief Single crossover operator.
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
	/// @copydoc crossover_op::run
	virtual void run(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const;
};

/// @brief Multiple crossover operator.
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
	/// @copydoc crossover_op::run
	virtual void run(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const;
};

////////////////////////////////////////////////////////////////////////////////
// mutation operators
////////////////////////////////////////////////////////////////////////////////

/// @brief Base class for mutation operator.
class mutation_op
{
public:
	/// @brief Destructor.
	virtual ~mutation_op() {}

	/// @brief Runs operator.
	/// @param limits genome limits
	/// @param population population genomes
	virtual void run(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const = 0;
};

/// @brief Uniform mutation operator.
///        Each individual is selected for mutation with probability 'p'.
///        Then random parameter is selected and mutated by replacing with value
///        from uniform distribution U(limit[i][0], limit[i][1])
///
class mutation_op_uniform : public mutation_op
{
private:
	const double p;
public:
	/// @brief Constructor.
	/// @param p probability of mutation of one individual
	explicit mutation_op_uniform(double p = 0.01) : p(p) {};

	/// @copydoc mutation_op::run
	virtual void run(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const;
};

/// @brief Normal mutation operator.
///        Each individual is selected for mutation with probability 'p'.
///        Then random parameter is selected and mutated by adding a value
///        from normal distribution N(0, c * (limit[i][1] - limit[i][0]))
///
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

	/// @copydoc mutation_op::run
	virtual void run(const std::vector<std::vector<double>> &limits, std::vector<std::vector<double>> &population) const;
};

////////////////////////////////////////////////////////////////////////////////
// score scaler
////////////////////////////////////////////////////////////////////////////////

/// @brief Base class for score scaler.
class score_scaler
{
public:
	/// @brief Destructor.
	virtual ~score_scaler() {}

	/// @brief Runs scaler.
	/// @param scores population scores
	/// @param scores_scaled scaled population scores
	virtual void run(const std::vector<double> &scores, std::vector<double> &scores_scaled) const = 0;
};

/// @brief None scaler.
///        This scaler performs no scaling at all.
class score_scaler_none : public score_scaler
{
public:
	/// @copydoc score_scaler::run
	virtual void run(const std::vector<double> &scores, std::vector<double> &scores_scaled) const;
};

/// @brief Offset scaler.
///        This scaler just shifts the scores minimum to zero.
class score_scaler_offset : public score_scaler
{
public:
	/// @copydoc score_scaler::run
	virtual void run(const std::vector<double> &scores, std::vector<double> &scores_scaled) const;
};

/// @brief Linear scaler.
///        This scaler scales the scores to the interval [0,1]
class score_scaler_linear : public score_scaler
{
public:
	/// @copydoc score_scaler::run
	virtual void run(const std::vector<double> &scores, std::vector<double> &scores_scaled) const;
};

////////////////////////////////////////////////////////////////////////////////
// evaluator
////////////////////////////////////////////////////////////////////////////////

/// @brief Evaluator (fitness) interface.
class evaluator
{
public:
	/// @brief Runs evaluator.
	///        It computes score (fitness) for given individual genome.
	/// @param params individual genome
	/// @param score score
	/// @return @c true if score was computed successfully, otherwise @c false
	virtual bool run(const std::vector<double> &params, double &score) const = 0;
};

////////////////////////////////////////////////////////////////////////////////
// genetic algorithm
////////////////////////////////////////////////////////////////////////////////

/// @brief Genetic algorithm.
class ga
{
private:
	std::vector<std::vector<double>> limits;

	const selection_op *selection;
	const crossover_op *crossover;
	const mutation_op  *mutation;
	const score_scaler *scaler;

	size_t popsize;
	bool   elisize;

	size_t genmax;
	size_t convn;
	double convmax;

	size_t thnum;

public:
	/// @brief Constructor.
	ga();

	/// @brief Configure optimizer
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
	/// @param thnum number of running threads, if zero then the number will be determined automatically
	/// @return @c true if configuring was successful, otherwise @c false
	bool configure(const std::vector<std::vector<double>> &limits,
	               const selection_op *selection, const crossover_op *crossover, const mutation_op *mutation, const score_scaler *scaler,
	               const size_t &popsize, const size_t &elisize, const size_t &genmax, const size_t &convn, const double &convmax, const size_t &thnum);

	/// @brief Runs genetic algorithm.
	/// @param eval evaluator
	/// @param params best genome
	/// @param score best score
	/// @return @c true if genetic algorithm finished succsessfully, otherwise @c false
	virtual bool run(const evaluator &eval, std::vector<double> &params, double &score) const;

private:
	bool initialize_population(std::vector<std::vector<double>> &population) const;
	bool calculate_scores(const evaluator &eval, const std::vector<std::vector<double>> &population, std::vector<double> &scores, std::vector<double> &scores_scaled) const;
	bool calculate_scores_mt(const evaluator &eval, const std::vector<std::vector<double>> &population, std::vector<double> &scores, std::vector<double> &scores_scaled) const;
	void calculate_stats(const std::vector<double> &scores, std::vector<size_t> &i_scores, double &mean_score, double &median_score) const;
	void calculate_convergence(double &conv, std::queue<double> &best_scores, const double &best_score) const;
};

} // namespace gann

#endif // GA_H

