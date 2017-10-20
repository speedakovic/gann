#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <ostream>

namespace gann
{

template<typename T>
std::ostream& operator<<(std::ostream &out, const std::vector<T> &x)
{
	out << '[';
	for (size_t i = 0; i < x.size(); ++i)
		out << x[i] << (i < x.size() - 1 ? ", " : "");
	out << "]";
	return out;
}

} // namespace gann

#endif // UTIL_H

