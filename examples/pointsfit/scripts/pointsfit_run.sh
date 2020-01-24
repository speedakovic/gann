#!/bin/sh

./pointsfit && gnuplot -p pointsfit_stats.gpi && gnuplot -p pointsfit_func.gpi

