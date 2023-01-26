#include <fantom/dataset.hpp>
#include <Eigen/Eigenvalues>

std::vector<int> compareGradients(std::vector<std::valarray<double>> gradients);

bool compareEigenvalues(Eigen::Vector2cd eigenValues);

Point2 getEdgeCenter2D(const ValueArray<Point2>& gridPoints, Cell& cell, int edge);

Point2 getCellCenter2D(const ValueArray<Point2>& gridPoints, Cell& cell);
