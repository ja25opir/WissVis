#include <fantom/algorithm.hpp>
#include <fantom/dataset.hpp>
#include <fantom/graphics.hpp>
#include <fantom/register.hpp>
#include <math.h>
#include<valarray>

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

        bool compareGradients(std::vector<std::valarray<double>> gradients)
        {
            /*  Quad
               0----3
               |    |
               |    |
               1----2
            */

            if(signbit(gradients[0][0]) != signbit(gradients[1][0]) || signbit(gradients[0][1]) != signbit(gradients[1][1]))
            {
                return true;
            }
            else if(signbit(gradients[0][0]) != signbit(gradients[3][0]) || signbit(gradients[0][1]) != signbit(gradients[3][1]))
            {
                return true;
            }
            else if(signbit(gradients[1][0]) != signbit(gradients[2][0]) || signbit(gradients[1][1]) != signbit(gradients[2][1]))
            {
                return true;
            }
            else if(signbit(gradients[2][0]) != signbit(gradients[3][0]) || signbit(gradients[2][1]) != signbit(gradients[3][1]))
            {
                return true;
            }
            else
            {
                return false;
            }

        }

        bool isInterestingCell(const ValueArray<Point2>& gridPoints, Cell& cell, const ValueArray<Scalar>& fieldValues, std::shared_ptr<const Field<2, Scalar>> field)
        {
            double epsilon = 0.1;

            std::valarray<double> gradientX;
            std::valarray<double> gradientY;
            std::valarray<double> baseVectorX = {1,0};
            std::valarray<double> baseVectorY = {0,1};

            std::valarray<double> gradientCombined;
            std::vector<std::valarray<double>> gradientVector;

            auto evaluator = field->makeEvaluator();

            for(size_t i = 0; i < cell.numVertices(); ++i)
            {
                //TODO Randbetrachtung
                Point2 evaluatorPointX = gridPoints[cell.index(i)];
                Point2 evaluatorPointY = evaluatorPointX;

                evaluatorPointX[0] += epsilon; //first in x direction

                if(evaluator->reset(evaluatorPointX, 0))
                {
                    auto valueX = evaluator->value();
                    gradientX = ((valueX[0] - fieldValues[cell.index(i)][0]) / epsilon) * baseVectorX;
                    //gradientsX.push_back((valueX[0] - fieldValues[cell.index(i)][0]) / epsilon);

                    //infoLog() << "grid point X: " << evaluatorPointX << std::endl;
                    //infoLog() << "eval value: " << valueX[0] << std::endl;
                    //infoLog() << "gradient: " << gradientsX.back() << std::endl;
                }
                else
                {
                    evaluatorPointX[0] -= 2*epsilon;

                    if(evaluator->reset(evaluatorPointX, 0))
                    {
                        auto valueX = evaluator->value();
                        gradientX = ((fieldValues[cell.index(i)][0] - valueX[0]) / epsilon) * baseVectorX;
                    }
                    else
                    {
                        infoLog() << "outside domain" << std::endl;
                    }
                }

                evaluatorPointY[1] += epsilon; //then in y direction

                if(evaluator->reset(evaluatorPointY, 0))
                {
                    auto valueY = evaluator->value();
                    gradientY = ((valueY[0] - fieldValues[cell.index(i)][0]) / epsilon) * baseVectorY;
                    //gradientsY.push_back((valueY[0] - fieldValues[cell.index(i)][0]) / epsilon);

                    //infoLog() << "grid point Y: " << evaluatorPointY << std::endl;
                    //infoLog() << "eval value: " << valueY[0] << std::endl;
                    //infoLog() << "gradient: " << gradientsY.back() << std::endl;
                }
                else
                {
                    evaluatorPointY[1] -= 2*epsilon;

                    if(evaluator->reset(evaluatorPointY, 0))
                    {
                        auto valueY = evaluator->value();
                        gradientY = ((fieldValues[cell.index(i)][0] - valueY[0]) / epsilon) * baseVectorY;
                    }
                    else
                    {
                        infoLog() << "outside domain" << std::endl;
                    }
                }

                gradientCombined = gradientX + gradientY;
                gradientVector.push_back(gradientCombined);
                //infoLog() << "gradient: " << gradientCombined[0] << "; " << gradientCombined[1] << std::endl;
            }

            if(!gradientVector.empty())
            {
                return compareGradients(gradientVector);
            }
            else
            {
                infoLog() << "empty list" << std::endl;
                return false;
            }
        }

        bool isExtrema()
        {
            //TODO:
            //  - Hesse Matrix in Zentrum von Zelle berechnen
            //  - auf negative Definitheit in diesem Punkt prüfen
        }


        virtual void execute( const Algorithm::Options& options, const volatile bool& /*abortFlag*/ ) override
        {
            std::shared_ptr<const Function<Scalar>> cFunction2D = options.get<Function<Scalar>>("Field_Cellbased2D");
            std::shared_ptr<const Function<Scalar>> pFunction2D = options.get<Function<Scalar>>("Field_Pointbased2D");
            std::shared_ptr<const Field<2, Scalar>> pField2D = options.get<Field<2, Scalar>>("Field_Pointbased2D");

            std::shared_ptr<const Function<Scalar>> cFunction3D = options.get<Function<Scalar>>("Field_Cellbased3D");
            std::shared_ptr<const Function<Scalar>> pFunction3D = options.get<Function<Scalar>>("Field_Pointbased3D");


            if(!cFunction2D && !pFunction2D && !cFunction3D && !pFunction3D)
            {
                infoLog() << "No input field!" << std::endl;
                return;
            }

            if(pFunction2D && cFunction2D)
            {
                std::shared_ptr<const Grid<2>> cGrid2D = std::dynamic_pointer_cast< const Grid<2>>(cFunction2D->domain());
                const ValueArray<Scalar>& cFieldValues2D = cFunction2D->values();
                const ValueArray<Point2>& cGridPoints2D = cGrid2D->points();

                std::shared_ptr<const Grid<2>> pGrid2D = std::dynamic_pointer_cast< const Grid<2>>(pFunction2D->domain());
                const ValueArray<Scalar>& pFieldValues2D = pFunction2D->values();
                const ValueArray<Point2>& pGridPoints2D = pGrid2D->points();
                PointSetBase::BoundingBox pBoundingBox2D = pGrid2D->getBoundingBox();

                //const ValueArray<Cell>& pGridCells2D = pGrid2D->cells();

                std::vector<Cell> interestingCells;
                std::vector<int> interestingCellsIndices;
                std::vector<int> extremaCellsIndices;


                for(size_t i = 0; i < pGrid2D->numCells(); ++i)
                {
                    Cell cell = pGrid2D->cell(i);
                    if(isInterestingCell(pGridPoints2D, cell, pFieldValues2D, pField2D))
                    {
                        //infoLog() << "------------------found interesting cell at: " << i << std::endl;
                        interestingCells.push_back(cell);
                        interestingCellsIndices.push_back(i);
                    }
                }
                infoLog() << "finished" << std::endl;

                for(size_t j = 0; j < interestingCellsIndices.size(); ++j)
                {
                    //infoLog() << "cell indices: " << interestingCellsIndices[j] << std::endl;
                    if(isExtrema())
                    {
                        //infoLog() << "------------------found extrema cell at: " << interestingCellsIndices[j] << std::endl;
                        extremaCellsIndices.push_back(interestingCellsIndices[j]);
                    }
                }

                setResult("RidgesAndValleys 2D", std::shared_ptr<const Grid<2>>(pGrid2D));
            }
            else
            {
                infoLog() << "Missing field input!" << std::endl;

            }

            if(pFunction3D && cFunction3D)
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

            }

        }
    };

    AlgorithmRegister <RidgesAndValleys> dummy("Hauptaufgabe/RidgesAndValleys", "Visualize Ridges and Valleys.");
}
