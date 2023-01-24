#include <fantom/algorithm.hpp>
#include <fantom/dataset.hpp>
#include <fantom/graphics.hpp>
#include <fantom/register.hpp>
#include <math.h>
#include <valarray>
#include <map>

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

                add<double>("Epsilon", "Epsilon value for gradient calculation", 1e-4);
            }
        };

        struct DataOutputs : public DataAlgorithm::DataOutputs
        {
            DataOutputs(fantom::DataOutputs::Control& control)
                : DataAlgorithm::DataOutputs(control)
            {
                add <const Grid<2>> ("RidgesAndValleys 2D");
                add <const Grid<3>> ("RidgesAndValleys 3D");
            }
        };

        RidgesAndValleys (InitData& data)
            : DataAlgorithm (data)
        {
        }

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
         * @brief getPartialGradient
         * @param evaluatorPoint - point coordinates
         * @param pointValue - point scalar value
         * @param fieldValues
         * @param evaluator
         * @param baseVector
         * @return
         */
        std::valarray<double> getPartialGradient(Point2 evaluatorPoint, double pointValue, std::unique_ptr< FieldEvaluator< 2UL, Tensor<double> > >& evaluator, std::valarray<double> baseVector, double epsilon) {
            //double epsilon = 1e-4;
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

        std::vector<Point2> isInterestingCell(const ValueArray<Point2>& gridPoints, Cell& cell, const ValueArray<Scalar>& fieldValues, std::shared_ptr<const Field<2, Scalar>> field, double epsilon)
        {
            std::valarray<double> gradientX;
            std::valarray<double> gradientY;
            std::valarray<double> baseVectorX = {1,0};
            std::valarray<double> baseVectorY = {0,1};

            std::valarray<double> gradientCombined;
            std::vector<std::valarray<double>> gradientVector;
            std::vector<Point2> edgeCenters;

            auto evaluator = field->makeEvaluator();

            for(size_t i = 0; i < cell.numVertices(); ++i)
            {
                Point2 point = gridPoints[cell.index(i)];
                double pointVal = fieldValues[cell.index(i)][0];
                gradientX = getPartialGradient(point, pointVal, evaluator, baseVectorX, epsilon);
                gradientY = getPartialGradient(point, pointVal, evaluator, baseVectorY, epsilon);

                gradientCombined = gradientX + gradientY;
                gradientVector.push_back(gradientCombined);
                //infoLog() << "gradient: " << gradientCombined[0] << "; " << gradientCombined[1] << std::endl;
            }

            if(!gradientVector.empty())
            {
                std::vector<int> edges = compareGradients(gradientVector);
                if(!edges.empty())
                {
                    for(size_t j = 0; j < edges.size(); ++j)
                    {
                        edgeCenters.push_back(getEdgeCenter2D(gridPoints, cell, edges[j]));
                    }
                }
            }
            return edgeCenters;
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

        bool isMaximum(const ValueArray<Point2>& gridPoints, Cell& cell, std::shared_ptr<const Field<2, Scalar>> field, double epsilon)
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

            auto evaluator = field->makeEvaluator();

            if(evaluator->reset(center, 0))
            {
                double centerVal = evaluator->value()[0];
                gradientX = getPartialGradient(center, centerVal, evaluator, baseVectorX, epsilon);
                gradientY = getPartialGradient(center, centerVal, evaluator, baseVectorY, epsilon);

                gradientXX = getPartialGradient(gradPointX, gradientX[0], evaluator, baseVectorX, epsilon);
                gradientYX = getPartialGradient(gradPointX, gradientY[0], evaluator, baseVectorX, epsilon);

                gradientXY = getPartialGradient(gradPointY, gradientX[0], evaluator, baseVectorY, epsilon);
                gradientYY = getPartialGradient(gradPointY, gradientY[0], evaluator, baseVectorY, epsilon);

                std::vector<std::valarray<double>> lineVector1 = {gradientXX, gradientXY};
                std::vector<std::valarray<double>> lineVector2 = {gradientYX, gradientYY};

                std::vector<std::vector<std::valarray<double>>> hesseMatrix = {lineVector1, lineVector2};

                return true;
            }

            //TODO:
            //  - auf negative Definitheit in diesem Punkt prüfen
        }


        virtual void execute( const Algorithm::Options& options, const volatile bool& /*abortFlag*/ ) override
        {
            std::shared_ptr<const Function<Scalar>> cFunction2D = options.get<Function<Scalar>>("Field_Cellbased2D");
            std::shared_ptr<const Function<Scalar>> pFunction2D = options.get<Function<Scalar>>("Field_Pointbased2D");
            std::shared_ptr<const Field<2, Scalar>> pField2D = options.get<Field<2, Scalar>>("Field_Pointbased2D");

            std::shared_ptr<const Function<Scalar>> cFunction3D = options.get<Function<Scalar>>("Field_Cellbased3D");
            std::shared_ptr<const Function<Scalar>> pFunction3D = options.get<Function<Scalar>>("Field_Pointbased3D");

            double epsilon = options.get<double>("Epsilon");


            if(!cFunction2D && !pFunction2D && !cFunction3D && !pFunction3D)
            {
                infoLog() << "No input field!" << std::endl;
                return;
            }

            if(pFunction2D)// && cFunction2D)
            {
                //std::shared_ptr<const Grid<2>> cGrid2D = std::dynamic_pointer_cast< const Grid<2>>(cFunction2D->domain());
                //const ValueArray<Scalar>& cFieldValues2D = cFunction2D->values();
                //const ValueArray<Point2>& cGridPoints2D = cGrid2D->points();

                std::shared_ptr<const Grid<2>> pGrid2D = std::dynamic_pointer_cast< const Grid<2>>(pFunction2D->domain());
                const ValueArray<Scalar>& pFieldValues2D = pFunction2D->values();
                const ValueArray<Point2>& pGridPoints2D = pGrid2D->points();
                //PointSetBase::BoundingBox pBoundingBox2D = pGrid2D->getBoundingBox();

                //const ValueArray<Cell>& pGridCells2D = pGrid2D->cells();

                std::vector<Cell> interestingCells;
                std::vector<int> interestingCellsIndices;
                std::map<int, std::vector<Point2>> ridgeValleyMap;
                std::vector<int> maximumCellsIndices;

                for(size_t i = 0; i < pGrid2D->numCells(); ++i)
                {
                    Cell cell = pGrid2D->cell(i);
                    std::vector<Point2> edgePoints = isInterestingCell(pGridPoints2D, cell, pFieldValues2D, pField2D, epsilon);

                    if(!edgePoints.empty())
                    {
                        //infoLog() << "------------------found interesting cell at: " << i << std::endl;
                        interestingCells.push_back(cell);
                        interestingCellsIndices.push_back(i);
                        ridgeValleyMap.insert({i, edgePoints});
                    }
                }
                infoLog() << "interesting cells found: ";
                infoLog() << ridgeValleyMap.size() << std::endl;
                //infoLog() << ridgeValleyMap[interestingCellsIndices[0]].size() << std::endl;

                //infoLog() << interestingCells.size() << std::endl;


                for(size_t j = 0; j < interestingCells.size(); ++j)
                {
                    //infoLog() << "cell indices: " << interestingCellsIndices[j] << std::endl;
                    if(isMaximum(pGridPoints2D, interestingCells[j], pField2D, epsilon))
                    {
                        //infoLog() << "------------------found extrema cell at: " << interestingCellsIndices[j] << std::endl;
                        maximumCellsIndices.push_back(interestingCellsIndices[j]);
                    }
                }

                setResult("RidgesAndValleys 2D", std::shared_ptr<const Grid<2>>(pGrid2D));
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
