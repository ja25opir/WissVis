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
                add<Field<2,Scalar>>( "Field_Cellbased2D", "A 2D cell based scalar field", definedOn<Grid<2>>(Grid<2>::Cells));
                add<Field<2,Scalar>>( "Field_Pointbased2D", "A 2D point based scalar field", definedOn<Grid<2>>(Grid<2>::Points));

                add<Field<3,Scalar>>( "Field_Cellbased3D", "A 3D cell based scalar field", definedOn<Grid<3>>(Grid<3>::Cells));
                add<Field<3,Scalar>>( "Field_Pointbased3D", "A 3D point basedscalar field", definedOn<Grid<3>>(Grid<3>::Points));

                add<double>("Epsilon", "Epsilon value for gradient calculation", 1e-3);

                add<double>("z_Scale", "Factor to visualize scalar values on z-Axis", 100);
            }
        };


        struct DataOutputs : public DataAlgorithm::DataOutputs
        {
            DataOutputs(fantom::DataOutputs::Control& control)
                : DataAlgorithm::DataOutputs(control)
            {
                add <LineSet<3>> ("RidgesAndValleys 2D");
                //add <const Grid<3>> ("RidgesAndValleys 3D");
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
        std::valarray<double> getPartialGradient(Point2 evaluatorPoint, double pointValue, std::unique_ptr< FieldEvaluator< 2UL, Tensor<double> > >& evaluator, std::valarray<double> baseVector, double epsilon) {
            std::valarray<double> gradient;

            Point2 baseVectorP2 = {baseVector[0], baseVector[1]};

            evaluatorPoint += epsilon * baseVectorP2;

            if(evaluator->reset(evaluatorPoint, 0))
            {
                auto value = evaluator->value();
                gradient = ((value[0] - pointValue) / epsilon) * baseVector;
            }
            else
            {
                evaluatorPoint -= 2 * epsilon * baseVectorP2;

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

        std::vector<Point3> isInterestingCell(const ValueArray<Point2>& gridPoints, Cell& cell, const ValueArray<Scalar>& fieldValues, std::shared_ptr<const Field<2, Scalar>> field, double epsilon, double zValScale)
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
                //double pointVal = fieldValues[cell.index(i)][0];
                if(evaluator->reset(point, 0))
                {
                    double pointVal = evaluator->value()[0];
                    gradientX = getPartialGradient(point, pointVal, evaluator, baseVectorX, epsilon);
                    gradientY = getPartialGradient(point, pointVal, evaluator, baseVectorY, epsilon);
                }

                gradientCombined = gradientX + gradientY;
                gradientVector.push_back(gradientCombined);
                //infoLog() << "gradient: " << gradientCombined[0] << "; " << gradientCombined[1] << std::endl;
            }

            if(!gradientVector.empty())
            {
                std::vector<int> edges = compareGradients(gradientVector);
                if(!edges.empty())
                {
                    if(isMaximum(gridPoints, cell, field, epsilon))
                    {
                        for(size_t j = 0; j < edges.size(); ++j)
                        {
                            Point2 edgeCenter2D = getEdgeCenter2D(gridPoints, cell, edges[j]);
                            double zVal = 0;
                            if(evaluator->reset(edgeCenter2D, 0)){
                               zVal = evaluator->value()[0] * zValScale;
                            }
                            Point3 edgeCenter3D = {edgeCenter2D[0], edgeCenter2D[1], zVal};
                            edgeCenters.push_back(edgeCenter3D);
                        }
                    }
                }
            }
            return edgeCenters;
        }

        Eigen::Matrix2d getHessianMatrix(std::valarray<double> evaluatorPoint, std::unique_ptr< FieldEvaluator< 2UL, Tensor<double> > >& evaluator, double epsilon, size_t dimension)
        {
            double f1, f2, f3, f4;

            Eigen::Matrix2d hessianMatrix(dimension,dimension);
            std::valarray<double> hessianArray(4);

            for(size_t i = 0; i < dimension; ++i)
            {
                for (size_t j = 0; j < dimension; ++j)
                {
                    std::valarray<double> baseVectorX = {0,0};
                    std::valarray<double> baseVectorY = {0,0};
                    baseVectorX[i] = 1;
                    baseVectorY[j] = 1;

                    std::valarray<double> f1Array = evaluatorPoint + (epsilon*baseVectorX) + (epsilon*baseVectorY);
                    Point2 f1Point = {f1Array[0], f1Array[1]};

                    if(evaluator->reset(f1Point, 0))
                    {
                        f1 = evaluator->value()[0];
                    }

                    std::valarray<double> f2Array = evaluatorPoint + (epsilon*baseVectorX) - (epsilon*baseVectorY);
                    Point2 f2Point = {f2Array[0], f2Array[1]};

                    if(evaluator->reset(f2Point, 0))
                    {
                        f2 = evaluator->value()[0];
                    }

                    std::valarray<double> f3Array = evaluatorPoint - (epsilon*baseVectorX) + (epsilon*baseVectorY);
                    Point2 f3Point = {f3Array[0], f3Array[1]};

                    if(evaluator->reset(f3Point, 0))
                    {
                        f3 = evaluator->value()[0];
                    }

                    std::valarray<double> f4Array = evaluatorPoint - (epsilon*baseVectorX) - (epsilon*baseVectorY);
                    Point2 f4Point = {f4Array[0], f4Array[1]};

                    if(evaluator->reset(f4Point, 0))
                    {
                        f4 = evaluator->value()[0];
                    }

                    double secondDerivative = (f1 - f2 - f3 + f4) / (4 * pow(epsilon,2));
                    hessianArray[i+j] = secondDerivative;
                }
            }
            hessianMatrix << hessianArray[0], hessianArray[1], hessianArray[2], hessianArray[3];
            return hessianMatrix;
        }



        bool isMaximum(const ValueArray<Point2>& gridPoints, Cell& cell, std::shared_ptr<const Field<2, Scalar>> field, double epsilon)
        {
            std::valarray<double> baseVectorX = {1,0};
            std::valarray<double> baseVectorY = {0,1};

            Point2 center = getCellCenter2D(gridPoints, cell);
            std::valarray<double> centerArray = {center[0], center[1]};

            auto evaluator = field->makeEvaluator();

            Eigen::Matrix2d hessianMatrix = getHessianMatrix(centerArray, evaluator, epsilon, 2);


            Eigen::Vector2cd eigenValues = hessianMatrix.eigenvalues();
            //infoLog() << hesseMatrixEigen << std::endl;

            if(compareEigenvalues(eigenValues))
            {
                return true;
            }
            return false;
        }

        virtual void execute( const Algorithm::Options& options, const volatile bool& /*abortFlag*/ ) override
        {
            std::shared_ptr<const Function<Scalar>> cFunction2D = options.get<Function<Scalar>>("Field_Cellbased2D");
            std::shared_ptr<const Function<Scalar>> pFunction2D = options.get<Function<Scalar>>("Field_Pointbased2D");
            std::shared_ptr<const Field<2, Scalar>> pField2D = options.get<Field<2, Scalar>>("Field_Pointbased2D");

            std::shared_ptr<const Function<Scalar>> cFunction3D = options.get<Function<Scalar>>("Field_Cellbased3D");
            std::shared_ptr<const Function<Scalar>> pFunction3D = options.get<Function<Scalar>>("Field_Pointbased3D");

            double epsilon = options.get<double>("Epsilon");

            double zScale = options.get<double>("z_Scale");

            if(!cFunction2D && !pFunction2D && !cFunction3D && !pFunction3D)
            {
                infoLog() << "No input field!" << std::endl;
                return;
            }

            if(pFunction2D)
            {
                std::shared_ptr<const Grid<2>> pGrid2D = std::dynamic_pointer_cast< const Grid<2>>(pFunction2D->domain());
                const ValueArray<Scalar>& pFieldValues2D = pFunction2D->values();
                //infoLog() << "EigenValues: " << eigenValues << std::endl;
                /*
                infoLog() << "gradientXX: " <<  gradientXX[0] << ", " << gradientXX[1] <<std::endl;
                infoLog() << "gradientXY: " <<  gradientXY[0] << ", " << gradientXY[1] <<std::endl;
                infoLog() << "gradientYX: " <<  gradientYX[0] << ", " << gradientYX[1] <<std::endl;
                infoLog() << "gradientYY: " <<  gradientYY[0] << ", " << gradientYY[1] <<std::endl<<std::endl;
                */
                const ValueArray<Point2>& pGridPoints2D = pGrid2D->points();

                LineSet<3> ridgeSet;
                std::vector<size_t> cellLineIndices;
                size_t lineIndex = 0;

                for(size_t i = 0; i < pGrid2D->numCells(); ++i)
                {
                    Cell cell = pGrid2D->cell(i);
                    std::vector<Point3> edgePoints = isInterestingCell(pGridPoints2D, cell, pFieldValues2D, pField2D, epsilon, zScale);

                    if(!edgePoints.empty())
                    {
                        for (size_t j = 0; j < edgePoints.size(); ++j) {
                            cellLineIndices.push_back(lineIndex);
                            ridgeSet.addPoint(Point3(edgePoints[j]));
                            ++lineIndex;
                        }
                        ridgeSet.addLine(cellLineIndices);
                        cellLineIndices.clear();
                    }

                }
                infoLog() << "ridgeSet num points: " << ridgeSet.numPoints() <<std::endl;
                infoLog() << "ridgeSet num lines: " << ridgeSet.numLines() <<std::endl;

                setResult("RidgesAndValleys 2D", std::make_shared<LineSet<3>>(ridgeSet));
            }
            else
            {
                infoLog() << "Missing field input!" << std::endl;

            }

            /*if(pFunction3D && cFunction3D)
            {
                std::shared_ptr<const Grid<3>> cGrid3D = std::dynamic_pointer_cast< const Grid<3>>(cFunction3D->domain());
                const ValueArray<Scalar>& cFieldValues3D = cFunction3D->values();
                const ValueArray<Point3>& cGridPoints3D = cGrid3D->points();

                std::shared_ptr<const Grid<3>> pGrid3D = std::dynamic_pointer_cast< const Grid<3>>(pFunction3D->domain());
                const ValueArray<Scalar>& pFieldValues3D = pFunction3D->values();
                const ValueArray<Point3>& pGridPoints3D = pGrid3D->points();

                setResult("RidgesAndValleys 3D", std::shared_ptr<const Grid<3>>(pGrid3D));
            }
            else
            {
                infoLog() << "Missing field input!" << std::endl;

            }*/

        }
    };

    AlgorithmRegister <RidgesAndValleys> dummy("Hauptaufgabe/RidgesAndValleys", "Visualize Ridges and Valleys.");
}
