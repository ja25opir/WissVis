#include <fantom/algorithm.hpp>
#include <fantom/dataset.hpp>
#include <fantom/graphics.hpp>
#include <fantom/register.hpp>
#include <math.h>

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

        bool isInterestingCell(const ValueArray<Point2>& gridPoints, Cell& cell, const ValueArray<Scalar>& fieldValues, std::shared_ptr<const Field<2, Scalar>> field)
        {
            double epsilon = 0.1;
            std::vector<double> gradientsX;
            std::vector<double> gradientsY;
            auto evaluator = field->makeEvaluator();
            //auto evaluator2 = field->makeDiscreteEvaluator();

            for(size_t i = 0; i < cell.numVertices(); ++i)
            {
                //TODO Randbetrachtung
                Point2 evaluatorPointX = gridPoints[cell.index(i)];
                Point2 evaluatorPointY = gridPoints[cell.index(i)];

                evaluatorPointX[0] += epsilon; //first in x direction

                if(evaluator->reset(evaluatorPointX, 0))
                {
                    auto valueX = evaluator->value();
                    gradientsX.push_back((valueX[0] - fieldValues[cell.index(i)][0]) / epsilon);

                    //infoLog() << "grid point X: " << evaluatorPointX << std::endl;
                    //infoLog() << "eval value: " << valueX[0] << std::endl;
                    //infoLog() << "gradient: " << gradientsX.back() << std::endl;
                }
                else
                {
                    infoLog() << "outside domain" << std::endl;
                }

                evaluatorPointY[1] += epsilon; //then in y direction

                if(evaluator->reset(evaluatorPointY, 0))
                {
                    auto valueY = evaluator->value();
                    gradientsY.push_back((valueY[0] - fieldValues[cell.index(i)][0]) / epsilon);

                    //infoLog() << "grid point Y: " << evaluatorPointY << std::endl;
                    //infoLog() << "eval value: " << valueY[0] << std::endl;
                    //infoLog() << "gradient: " << gradientsY.back() << std::endl;
                }
                else
                {
                    infoLog() << "outside domain" << std::endl;
                }
            }

            /*  Quad
               0----3
               |    |
               |    |
               1----2
            */

            if(signbit(gradientsX[0]) != signbit(gradientsX[3]) || signbit(gradientsX[1]) != signbit(gradientsX[2]) || signbit(gradientsY[0]) != signbit(gradientsY[1]) || signbit(gradientsY[2]) != signbit(gradientsY[3]))
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


            if(!cFunction2D && !pFunction2D && !cFunction3D && !pFunction3D)
            {
                infoLog() << "No input field!" << std::endl;
                return;
            }

            if(cFunction2D)
            {
                std::shared_ptr<const Grid<2>> cGrid2D = std::dynamic_pointer_cast< const Grid<2>>(cFunction2D->domain());
                const ValueArray<Scalar>& cFieldValues2D = cFunction2D->values();
                const ValueArray<Point2>& cGridPoints2D = cGrid2D->points();
            }

            if(pFunction2D)
            {
                std::shared_ptr<const Grid<2>> pGrid2D = std::dynamic_pointer_cast< const Grid<2>>(pFunction2D->domain());
                const ValueArray<Scalar>& pFieldValues2D = pFunction2D->values();
                const ValueArray<Point2>& pGridPoints2D = pGrid2D->points();
                //const ValueArray<Cell>& pGridCells2D = pGrid2D->cells();

                std::vector<Cell> interestingCell;


                for(size_t i = 0; i < pGrid2D->numCells(); ++i)
                {
                    Cell cell = pGrid2D->cell(i);
                    if(isInterestingCell(pGridPoints2D, cell, pFieldValues2D, pField2D))
                    {
                        infoLog() << "------------------found interesting cell at: " << i << std::endl;
                        interestingCell.push_back(cell);
                    }
                }

                setResult("RidgesAndValleys 2D", std::shared_ptr<const Grid<2>>(pGrid2D));
            }

            if(cFunction3D)
            {
                std::shared_ptr<const Grid<3>> cGrid3D = std::dynamic_pointer_cast< const Grid<3>>(cFunction3D->domain());
                const ValueArray<Scalar>& cFieldValues3D = cFunction3D->values();
                const ValueArray<Point3>& cGridPoints3D = cGrid3D->points();
            }

            if(pFunction3D)
            {
                std::shared_ptr<const Grid<3>> pGrid3D = std::dynamic_pointer_cast< const Grid<3>>(pFunction3D->domain());
                const ValueArray<Scalar>& pFieldValues3D = pFunction3D->values();
                const ValueArray<Point3>& pGridPoints3D = pGrid3D->points();

                setResult("RidgesAndValleys 3D", std::shared_ptr<const Grid<3>>(pGrid3D));

            }

        }
    };

    AlgorithmRegister <RidgesAndValleys> dummy("Hauptaufgabe/RidgesAndValleys", "Visualize Ridges and Valleys.");
}
