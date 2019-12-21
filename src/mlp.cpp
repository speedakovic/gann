#include <gann/mlp.h>
#include <gann/log.h>
#include <gann/util.h>

#include <cmath>

// network             <=> vector of layers
// layer               <=> vector of neurons
// neuron              <=> pair of vector of weights and activation function
// weight              <=> floating point number
// activation function <=> pointer to activation function

namespace gann
{

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

bool mlp::set_architecture(const std::vector<size_t> &arch)
{
	if (arch.size() < 2) {
		GANN_ERR("too few items in mlp architecture descriptor" << std::endl);
		return false;
	}

	for (const auto &i : arch)
		if (i == 0) {
			GANN_ERR("zero item in mlp architecture descriptor" << std::endl);
			return false;
		}

	network.resize(arch.size() - 1);

	for (size_t i = 1; i < arch.size(); ++i)
		network[i - 1] = std::vector<std::pair<std::vector<double>, activation_function>>(arch[i], std::pair<std::vector<double>,activation_function>(std::vector<double>(arch[i - 1] + 1), af_identity));

	return true;
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

bool mlp::set_activation_functions(const activation_function &af)
{
	for (auto &layer : network)
		for (auto &neuron : layer)
			neuron.second = af;

	return true;
}

bool mlp::set_activation_functions(const std::vector<activation_function> &af)
{
	size_t i = 0;
	for (const auto &layer : network)
		i += layer.size();

	if (af.size() != i) {
		GANN_ERR("number of activation functions doesn't match the number of neurons" << std::endl);
		return false;
	}

	i = 0;
	for (auto &layer : network)
		for (auto &neuron : layer)
			neuron.second = af[i++];

	return true;
}

bool mlp::set_activation_functions_by_layers(const std::vector<activation_function> &af)
{
	if (af.size() != network.size()) {
		GANN_ERR("number of activation functions doesn't match the number of layers" << std::endl);
		return false;
	}

	for (size_t i = 0; i < af.size(); ++i)
		for (auto &neuron : network[i])
			neuron.second = af[i];

	return true;
}

bool mlp::set_weights(const std::vector<double> &weights)
{
	size_t i = 0;
	for (const auto &layer : network)
		for (const auto &neuron : layer)
			i += neuron.first.size();

	if (weights.size() != i) {
		GANN_ERR("number of weights doesn't match" << std::endl);
		return false;
	}

	i = 0;
	for (auto &layer : network)
		for (auto &neuron : layer)
			for (auto &weight : neuron.first)
				weight = weights[i++];

	return true;
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

std::vector<double> mlp::propagate(const std::vector<double> &in)
{
	if (network.empty()) {
		GANN_ERR("empty network" << std::endl);
		return {};
	}

	std::vector<double> inputs = in;
	std::vector<double> outputs;

	for (const auto &layer : network) {
		outputs.clear();
		for (const auto &neuron : layer) {

			if (inputs.size() != neuron.first.size() - 1) {
				GANN_ERR("number of inputs doesn't match the number of weights" << std::endl);
				return {};
			}

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

