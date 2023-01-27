#include <fantom/algorithm.hpp>
#include <fantom/dataset.hpp>
#include <fantom/graphics.hpp>
#include <fantom/register.hpp>
#include <math.h>
#include <Eigen/Eigenvalues>
#include <valarray>
#include <map>
#include "helpers.h"
#include <fantom-plugins/utils/Graphics/ObjectRenderer.hpp>

using namespace fantom;

namespace
{
    class RidgesAndValleys : public DataAlgorithm
    {

    public:
        struct Options : public DataAlgorithm::Options
        {
            Options( fantom::Options::Control& control )
                : DataAlgorithm::Options( control )
            {
                add<Field<2,Scalar>>( "Field_Pointbased2D", "A 2D point based scalar field", definedOn<Grid<2>>(Grid<2>::Points));

                add<Field<3,Scalar>>( "Field_Pointbased3D", "A 3D point basedscalar field", definedOn<Grid<3>>(Grid<3>::Points));

                add<double>("Epsilon", "Epsilon value for gradient calculation", 1e-3);
            }
        };


        struct DataOutputs : public DataAlgorithm::DataOutputs
        {
            DataOutputs(fantom::DataOutputs::Control& control)
                : DataAlgorithm::DataOutputs(control)
            {
                add <LineSet<3>> ("RidgesAndValleys 2D");
                add <LineSet<3>> ("RidgesAndValleys 3D");
            }
        };

        RidgesAndValleys (InitData& data)
            : DataAlgorithm (data)
        {
        }

        /**
         * @brief getPartialGradient (for 2D Points only!)
         * @param evaluatorPoint - point coordinates
         * @param pointValue - point scalar value
         * @param evaluator - field evaluator for interpolation
         * @param baseVector - 2D base vector
         * @param epsilon - stepsize for gradient calculation
         * @return
         */
        std::valarray<double> getPartialGradient2D(Point2 evaluatorPoint, double pointValue, std::unique_ptr< FieldEvaluator< 2UL, Tensor<double> > >& evaluator, std::valarray<double> baseVector, double epsilon) {
            std::valarray<double> gradient;

            Point2 baseVectorTensor;
            baseVectorTensor = {baseVector[0], baseVector[1]};

            evaluatorPoint += epsilon * baseVectorTensor;

            if(evaluator->reset(evaluatorPoint, 0))
            {
                auto value = evaluator->value();
                gradient = ((value[0] - pointValue) / epsilon) * baseVector;
            }
            else
            {
                evaluatorPoint -= 2 * epsilon * baseVectorTensor;

                if(evaluator->reset(evaluatorPoint, 0))
                {
                    auto value = evaluator->value();
                    gradient = ((pointValue - value[0]) / epsilon) * baseVector;
                }
                else
                {
                    infoLog() << "outside domain" << std::endl;
                }
            }
            return gradient;
        }

        std::vector<Point3> isInterestingCell2D(const ValueArray<Point2>& gridPoints, Cell& cell, const ValueArray<Scalar>& fieldValues, std::shared_ptr<const Field<2, Scalar>>& field, double epsilon)
        {
            std::valarray<double> gradientX;
            std::valarray<double> gradientY;
            std::valarray<double> baseVectorX = {1,0};
            std::valarray<double> baseVectorY = {0,1};

            std::valarray<double> gradientCombined;
            std::vector<std::valarray<double>> gradientVector;
            std::vector<Point3> edgeCenters;

            auto evaluator = field->makeEvaluator();

            for(size_t i = 0; i < cell.numVertices(); ++i)
            {
                Point2 point = gridPoints[cell.index(i)];
                double pointVal = fieldValues[cell.index(i)][0];
                gradientX = getPartialGradient2D(point, pointVal, evaluator, baseVectorX, epsilon);
                gradientY = getPartialGradient2D(point, pointVal, evaluator, baseVectorY, epsilon);

                gradientCombined = gradientX + gradientY;
                gradientVector.push_back(gradientCombined);
                //infoLog() << "gradient: " << gradientCombined[0] << "; " << gradientCombined[1] << std::endl;
            }

            if(!gradientVector.empty())
            {
                std::vector<int> edges = compareGradients2D(gradientVector);
                if(!edges.empty())
                {
                    if(isMaximum(gridPoints, cell, field, epsilon))
                    {
                        for(size_t j = 0; j < edges.size(); ++j)
                        {
                            Point2 edgeCenter2D = getEdgeCenter2D(gridPoints, cell, edges[j]);
                            Point3 edgeCenter3D = {edgeCenter2D[0], edgeCenter2D[1], 0};
                            edgeCenters.push_back(edgeCenter3D);
                        }
                    }

                }
            }
            return edgeCenters;
        }

        std::valarray<double> getPartialGradient3D(Point3 evaluatorPoint, double pointValue, std::unique_ptr< FieldEvaluator< 3UL, Tensor<double> > >& evaluator, std::valarray<double> baseVector, double epsilon) {
            std::valarray<double> gradient;

            Point3 baseVectorTensor;
            baseVectorTensor = {baseVector[0], baseVector[1], baseVector[2]};

            evaluatorPoint += epsilon * baseVectorTensor;

            if(evaluator->reset(evaluatorPoint, 0))
            {
                auto value = evaluator->value();
                gradient = ((value[0] - pointValue) / epsilon) * baseVector;
            }
            else
            {
                evaluatorPoint -= 2 * epsilon * baseVectorTensor;

                if(evaluator->reset(evaluatorPoint, 0))
                {
                    auto value = evaluator->value();
                    gradient = ((pointValue - value[0]) / epsilon) * baseVector;
                }
                else
                {
                    infoLog() << "outside domain" << std::endl;
                }
            }
            return gradient;
        }

        std::vector<Point3> isInterestingCell3D(const ValueArray<Point3>& gridPoints, Cell& cell, const ValueArray<Scalar>& fieldValues, std::shared_ptr<const Field<3, Scalar>>& field, double epsilon)
        {
            std::valarray<double> gradientX;
            std::valarray<double> gradientY;
            std::valarray<double> gradientZ;
            std::valarray<double> baseVectorX = {1,0,0};
            std::valarray<double> baseVectorY = {0,1,0};
            std::valarray<double> baseVectorZ = {0,0,1};

            std::valarray<double> gradientCombined;
            std::vector<std::valarray<double>> gradientVector;
            std::vector<Point3> edgeCenters;

            auto evaluator = field->makeEvaluator();

            for(size_t i = 0; i < cell.numVertices(); ++i)
            {
                Point3 point = gridPoints[cell.index(i)];
                double pointVal = fieldValues[cell.index(i)][0];
                gradientX = getPartialGradient3D(point, pointVal, evaluator, baseVectorX, epsilon);
                gradientY = getPartialGradient3D(point, pointVal, evaluator, baseVectorY, epsilon);
                gradientZ = getPartialGradient3D(point, pointVal, evaluator, baseVectorZ, epsilon);

                gradientCombined = gradientX + gradientY + gradientZ;
                gradientVector.push_back(gradientCombined);
                //infoLog() << "gradient: " << gradientCombined[0] << "; " << gradientCombined[1] << std::endl;
            }


            if(!gradientVector.empty())
            {
                std::vector<int> edges = compareGradients3D(gradientVector);
                if(!edges.empty())
                {
                    for(size_t j = 0; j < edges.size(); ++j)
                    {
                        Point3 edgeCenter3D = getEdgeCenter3D(gridPoints, cell, edges[j]);
                        edgeCenters.push_back(edgeCenter3D);
                    }
                    /*
                    if(isMaximum(gridPoints, cell, field, epsilon))
                    {
                        for(size_t j = 0; j < edges.size(); ++j)
                        {
                            Point2 edgeCenter2D = getEdgeCenter3D(gridPoints, cell, edges[j]);
                            Point3 edgeCenter3D = {edgeCenter2D[0], edgeCenter2D[1], 0};
                            edgeCenters.push_back(edgeCenter3D);
                        }
                    }*/

                }
            }
            return edgeCenters;
        }

        bool isMaximum(const ValueArray<Point2>& gridPoints, Cell& cell, std::shared_ptr<const Field<2, Scalar>>& field, double epsilon)
        {
            std::valarray<double> gradientX;
            std::valarray<double> gradientY;

            std::valarray<double> gradientXX;
            std::valarray<double> gradientYX;
            std::valarray<double> gradientXY;
            std::valarray<double> gradientYY;

            std::valarray<double> baseVectorX = {1,0};
            std::valarray<double> baseVectorY = {0,1};

            Point2 center = getCellCenter2D(gridPoints, cell);
            Point2 gradPointX = {center[0]+epsilon, center[1]};
            Point2 gradPointY = {center[0], center[1]+epsilon};

            /*
            infoLog() << "epsilon: " << epsilon << std::endl;
            infoLog() << "centerPoint: " << center << std::endl;
            infoLog() << "gradPointX: " << gradPointX << std::endl;
            infoLog() << "gradPointY: " << gradPointY << std::endl;
*/
            auto evaluator = field->makeEvaluator();

            if(evaluator->reset(center, 0))
            {
                double centerVal = evaluator->value()[0];
                gradientX = getPartialGradient2D(center, centerVal, evaluator, baseVectorX, epsilon);
                gradientY = getPartialGradient2D(center, centerVal, evaluator, baseVectorY, epsilon);
            }
            if(evaluator->reset(gradPointX,0))
            {
                gradientXX = getPartialGradient2D(gradPointX, evaluator->value()[0], evaluator, baseVectorX, epsilon);
                gradientXY = getPartialGradient2D(gradPointX, evaluator->value()[0], evaluator, baseVectorY, epsilon);
            }
            if(evaluator->reset(gradPointY, 0))
            {
                gradientYX = getPartialGradient2D(gradPointY, evaluator->value()[0], evaluator, baseVectorX, epsilon);
                gradientYY = getPartialGradient2D(gradPointY, evaluator->value()[0], evaluator, baseVectorY, epsilon);
            }

            //std::vector<std::valarray<double>> lineVector1 = {gradientXX, gradientXY};
            //std::vector<std::valarray<double>> lineVector2 = {gradientYX, gradientYY};
            //std::vector<std::vector<std::valarray<double>>> hesseMatrix = {lineVector1, lineVector2};

            Eigen::Matrix2d hesseMatrixEigen(2,2);
            hesseMatrixEigen << gradientXX[0], gradientXY[1], gradientYX[0], gradientYY[1];

            Eigen::Vector2cd eigenValues = hesseMatrixEigen.eigenvalues();

            if(compareEigenvalues(eigenValues))
            {
                return true;
            }

            //infoLog() << "EigenValues: " << eigenValues << std::endl;
            /*
            infoLog() << "gradientXX: " <<  gradientXX[0] << ", " << gradientXX[1] <<std::endl;
            infoLog() << "gradientXY: " <<  gradientXY[0] << ", " << gradientXY[1] <<std::endl;
            infoLog() << "gradientYX: " <<  gradientYX[0] << ", " << gradientYX[1] <<std::endl;
            infoLog() << "gradientYY: " <<  gradientYY[0] << ", " << gradientYY[1] <<std::endl<<std::endl;
            */
            return false;
        }

        virtual void execute( const Algorithm::Options& options, const volatile bool& /*abortFlag*/ ) override
        {
            std::shared_ptr<const Function<Scalar>> pFunction2D = options.get<Function<Scalar>>("Field_Pointbased2D");
            std::shared_ptr<const Field<2, Scalar>> pField2D = options.get<Field<2, Scalar>>("Field_Pointbased2D");

            std::shared_ptr<const Function<Scalar>> pFunction3D = options.get<Function<Scalar>>("Field_Pointbased3D");
            std::shared_ptr<const Field<3, Scalar>> pField3D = options.get<Field<3, Scalar>>("Field_Pointbased3D");


            double epsilon = options.get<double>("Epsilon");


            if(!pFunction2D && !pFunction3D)
            {
                infoLog() << "No input field!" << std::endl;
                return;
            }

            if(pFunction2D)
            {
                std::shared_ptr<const Grid<2>> pGrid2D = std::dynamic_pointer_cast< const Grid<2>>(pFunction2D->domain());
                const ValueArray<Scalar>& pFieldValues2D = pFunction2D->values();
                const ValueArray<Point2>& pGridPoints2D = pGrid2D->points();

                LineSet<3> ridgeSet2D;
                std::vector<size_t> cellLineIndices2D;
                size_t lineIndex2D = 0;

                for(size_t i = 0; i < pGrid2D->numCells(); ++i)
                {
                    Cell cell2D = pGrid2D->cell(i);
                    std::vector<Point3> edgePoints2D = isInterestingCell2D(pGridPoints2D, cell2D, pFieldValues2D, pField2D, epsilon);

                    if(!edgePoints2D.empty())
                    {
                        for (size_t j = 0; j < edgePoints2D.size(); ++j) {
                            cellLineIndices2D.push_back(lineIndex2D);
                            ridgeSet2D.addPoint(Point3(edgePoints2D[j]));
                            ++lineIndex2D;
                        }
                        ridgeSet2D.addLine(cellLineIndices2D);
                        cellLineIndices2D.clear();
                    }

                }
                infoLog() << "ridgeSet num points: " << ridgeSet2D.numPoints() <<std::endl;
                infoLog() << "ridgeSet num lines: " << ridgeSet2D.numLines() <<std::endl;

                setResult("RidgesAndValleys 2D", std::make_shared<LineSet<3>>(ridgeSet2D));
            }

            if(pFunction3D)
            {
                std::shared_ptr<const Grid<3>> pGrid3D = std::dynamic_pointer_cast< const Grid<3>>(pFunction3D->domain());
                const ValueArray<Scalar>& pFieldValues3D = pFunction3D->values();
                const ValueArray<Point3>& pGridPoints3D = pGrid3D->points();

                LineSet<3> ridgeSet3D;
                std::vector<size_t> cellLineIndices3D;
                size_t lineIndex3D = 0;

                for(size_t i = 0; i < pGrid3D->numCells(); ++i)
                {
                    Cell cell3D = pGrid3D->cell(i);
                    std::vector<Point3> edgePoints3D = isInterestingCell3D(pGridPoints3D, cell3D, pFieldValues3D, pField3D, epsilon);

                    if(!edgePoints3D.empty())
                    {
                        for (size_t j = 0; j < edgePoints3D.size(); ++j) {
                            cellLineIndices3D.push_back(lineIndex3D);
                            ridgeSet3D.addPoint(Point3(edgePoints3D[j]));
                            ++lineIndex3D;
                        }
                        ridgeSet3D.addLine(cellLineIndices3D);
                        cellLineIndices3D.clear();
                    }
                }

                setResult("RidgesAndValleys 3D", std::make_shared<LineSet<3>>(ridgeSet3D));
            }

        }
    };

    AlgorithmRegister <RidgesAndValleys> dummy("Hauptaufgabe/RidgesAndValleys", "Visualize Ridges and Valleys.");
}
