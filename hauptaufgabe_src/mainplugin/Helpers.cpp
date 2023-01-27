#include <fantom/dataset.hpp>
#include <math.h>
#include <valarray>
#include <Eigen/Eigenvalues>

using namespace fantom;


/**
 * @brief compareGradients - classifies type of extrema
 * @param gradients - array of gradients
 * @return edges - indicates edges between gradients with different signs
 */
std::vector<int> compareGradients(std::vector<std::valarray<double>> gradients)
{
    /*  Quad
       0----3
       |    |
       |    |
       1----2
    */

    std::vector<int> edges;

    if(signbit(gradients[0][0]) != signbit(gradients[1][0]) || signbit(gradients[0][1]) != signbit(gradients[1][1]))
    {
        edges.push_back(0);
    }
    if(signbit(gradients[1][0]) != signbit(gradients[2][0]) || signbit(gradients[1][1]) != signbit(gradients[2][1]))
    {
        edges.push_back(1);
    }
    if(signbit(gradients[2][0]) != signbit(gradients[3][0]) || signbit(gradients[2][1]) != signbit(gradients[3][1]))
    {
        edges.push_back(2);
    }
    if(signbit(gradients[0][0]) != signbit(gradients[3][0]) || signbit(gradients[0][1]) != signbit(gradients[3][1]))
    {
        edges.push_back(3);
    }
    return edges;
}

/**
 * @brief compareEigenvaluesMax - classifies maximum cells
 * @param eigenValues - eigenvalues of hessian matrix
 * @return bool
 */
bool compareEigenvaluesMax(Eigen::Vector2cd eigenValues)
{
    if(signbit(eigenValues(0).real()) && signbit(eigenValues(1).real()))
    {
        return true;
    }
    return false;
}

/**
 * @brief compareEigenvaluesMin - classifies minimum cells
 * @param eigenValues - eigenvalues of hessian matrix
 * @return bool
 */
bool compareEigenvaluesMin(Eigen::Vector2cd eigenValues)
{
    if(eigenValues(0).real() > 0 && eigenValues(1).real() > 0)
    {
        return true;
    }
    return false;
}

/**
 * @brief getEdgeCenter2D - computes center of edges
 * @param gridPoints - array of grid points
 * @param cell - grid cell
 * @param edge - cell edge
 * @return point - center point
 */
Point2 getEdgeCenter2D(const ValueArray<Point2>& gridPoints, Cell& cell, int edge)
{
    float sumX = 0;
    float sumY = 0;

    if(edge != 3)
    {
        sumX = gridPoints[cell.index(edge)][0] + gridPoints[cell.index(edge+1)][0];
        sumY = gridPoints[cell.index(edge)][1] + gridPoints[cell.index(edge+1)][1];
    }
    else
    {
        sumX = gridPoints[cell.index(edge)][0] + gridPoints[cell.index(0)][0];
        sumY = gridPoints[cell.index(edge)][1] + gridPoints[cell.index(0)][1];
    }

    return {sumX/2, sumY/2};
}

/**
 * @brief getCellCenter2D - computes center of cell
 * @param gridPoints - array of grid points
 * @param cell - grid cell
 * @return point - center point
 */
Point2 getCellCenter2D(const ValueArray<Point2>& gridPoints, Cell& cell){
    double sumX = 0;
    double sumY = 0;
    for(size_t i = 0; i < cell.numVertices(); ++i)
    {
        sumX += gridPoints[cell.index(i)][0];
        sumY += gridPoints[cell.index(i)][1];
    }
    return {sumX / 4, sumY / 4};
}
