#include <fantom/dataset.hpp>
#include <Eigen/Eigenvalues>

std::vector<int> compareGradients2D(std::vector<std::valarray<double>> gradients);

std::vector<int> compareGradients3D(std::vector<std::valarray<double>> gradients);


bool compareEigenvalues(Eigen::Vector2cd eigenValues);


Point2 getEdgeCenter2D(const ValueArray<Point2>& gridPoints, Cell& cell, int edge);

Point3 getEdgeCenter3D(const ValueArray<Point3>& gridPoints, Cell& cell, int edge);


Point2 getCellCenter2D(const ValueArray<Point2>& gridPoints, Cell& cell);

Point3 getCellCenter3D(const ValueArray<Point3>& gridPoints, Cell& cell);
