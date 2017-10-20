#ifndef MLP_H
#define MLP_H

#include <vector>
#include <utility>
#include <iostream>

namespace gann
{

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
	/// @return @c true if setting was successful, otherwise @c false
	bool set_architecture(const std::vector<size_t> &arch);

	/// @brief Gets network architecture.
	/// @return architecture descriptor
	std::vector<size_t> get_architecture() const;

	/// @brief Sets activation functions.
	/// @param af one activation function to be set to all neurons
	/// @return @c true if setting was successful, otherwise @c false
	bool set_activation_functions(const activation_function &af);

	/// @brief Sets activation functions.
	/// @param af activation functions to be set,
	///           number of activation functions must be equal number of neurons
	/// @return @c true if setting was successful, otherwise @c false
	bool set_activation_functions(const std::vector<activation_function> &af);

	/// @brief Sets activation functions.
	/// @param af activation functions to be set,
	///           number of activation functions must be equal number of layers
	/// @return @c true if setting was successful, otherwise @c false
	bool set_activation_functions_by_layers(const std::vector<activation_function> &af);

	/// @brief Sets weights.
	/// @param weights weights to be set,
	///                number of weights must be equal number of weights across all existing neurons
	/// @return @c true if setting was successful, otherwise @c false
	bool set_weights(const std::vector<double> &weights);

	/// @brief Gets weights.
	/// @return weights
	std::vector<double> get_weights() const;

	/// @brief Propagates input vector through network and returns resulting output vector.
	/// @param in input vector, its size must be equal number of inputs specified in network architecture
	/// @return output vector, its size equals number of neurons in last layer or it is empty if some error occurred
	std::vector<double> propagate(const std::vector<double> &in);

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

#endif // MLP_H

