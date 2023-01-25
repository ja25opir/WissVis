#include <fantom/dataset.hpp>

std::vector<int> compareGradients(std::vector<std::valarray<double>> gradients);

Point2 getEdgeCenter2D(const ValueArray<Point2>& gridPoints, Cell& cell, int edge);

Point2 getCellCenter2D(const ValueArray<Point2>& gridPoints, Cell& cell);
