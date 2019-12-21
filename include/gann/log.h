#ifndef LOG_H
#define LOG_H

#include <iostream>

#define GANN_DBG(x) do {std::cerr << x;} while(0);
#define GANN_ERR(x) do {std::cerr << x;} while(0);

#endif // LOG_H

