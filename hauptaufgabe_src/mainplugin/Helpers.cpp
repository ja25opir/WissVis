#include <fantom/dataset.hpp>
#include <math.h>
#include <valarray>
#include <Eigen/Eigenvalues>

using namespace fantom;


std::vector<int> compareGradients2D(std::vector<std::valarray<double>> gradients)
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

std::vector<int> compareGradients3D(std::vector<std::valarray<double>> gradients)
{
    std::vector<int> edges;

    if(signbit(gradients[0][0]) != signbit(gradients[1][0]) || signbit(gradients[0][1]) != signbit(gradients[1][1]) || signbit(gradients[0][2]) != signbit(gradients[1][2]))
    {
        edges.push_back(0);
    }
    if(signbit(gradients[1][0]) != signbit(gradients[2][0]) || signbit(gradients[1][1]) != signbit(gradients[2][1]) || signbit(gradients[1][2]) != signbit(gradients[2][2]))
    {
        edges.push_back(1);
    }
    if(signbit(gradients[2][0]) != signbit(gradients[3][0]) || signbit(gradients[2][1]) != signbit(gradients[3][1]) || signbit(gradients[2][2]) != signbit(gradients[3][2]))
    {
        edges.push_back(2);
    }
    if(signbit(gradients[0][0]) != signbit(gradients[3][0]) || signbit(gradients[0][1]) != signbit(gradients[3][1]) || signbit(gradients[0][2]) != signbit(gradients[3][2]))
    {
        edges.push_back(3);
    }
    return edges;
}


bool compareEigenvalues(Eigen::Vector2cd eigenValues)
{
    if(signbit(eigenValues(0).real()) && signbit(eigenValues(1).real()))
    {
        return true;
    }
    return false;
}


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

Point2 getEdgeCenter3D(const ValueArray<Point3>& gridPoints, Cell& cell, int edge)
{
    float sumX = 0;
    float sumY = 0;
    float sumZ = 0;

    if(edge != 3)
    {
        sumX = gridPoints[cell.index(edge)][0] + gridPoints[cell.index(edge+1)][0];
        sumY = gridPoints[cell.index(edge)][1] + gridPoints[cell.index(edge+1)][1];
        sumZ = gridPoints[cell.index(edge)][2] + gridPoints[cell.index(edge+1)][2];
    }
    else
    {
        sumX = gridPoints[cell.index(edge)][0] + gridPoints[cell.index(0)][0];
        sumY = gridPoints[cell.index(edge)][1] + gridPoints[cell.index(0)][1];
        sumZ = gridPoints[cell.index(edge)][2] + gridPoints[cell.index(0)][2];
    }

    return {sumX/2, sumY/2, sumZ/2};
}


Point2 getCellCenter2D(const ValueArray<Point2>& gridPoints, Cell& cell){
    double sumX = 0;
    double sumY = 0;
    for(size_t i = 0; i < cell.numVertices(); ++i)
    {
        sumX += gridPoints[cell.index(i)][0];
        sumY += gridPoints[cell.index(i)][1];
    }
    return {sumX/4, sumY/4};
}

Point2 getCellCenter3D(const ValueArray<Point3>& gridPoints, Cell& cell){
    double sumX = 0;
    double sumY = 0;
    double sumZ = 0;
    for(size_t i = 0; i < cell.numVertices(); ++i)
    {
        sumX += gridPoints[cell.index(i)][0];
        sumY += gridPoints[cell.index(i)][1];
        sumZ += gridPoints[cell.index(i)][2];
    }
    return {sumX/4, sumY/4, sumZ/4};
}
