#include <fantom/algorithm.hpp>
#include <fantom/dataset.hpp>
#include <fantom/graphics.hpp>
#include <fantom/register.hpp>

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
                add <LineSet<3>> ("RidgesAndValleys");
            }
        };

        RidgesAndValleys (InitData& data)
            : DataAlgorithm (data)
        {
        }

        virtual void execute( const Algorithm::Options& options, const volatile bool& /*abortFlag*/ ) override
        {
            std::shared_ptr<const Function<Scalar>> cellFunction2D = options.get<Function<Scalar>>("Field_Cellbased2D");
            std::shared_ptr<const Function<Scalar>> pointFunction2D = options.get<Function<Scalar>>("Field_Pointbased2D");

            std::shared_ptr<const Function<Scalar>> cellFunction3D = options.get<Function<Scalar>>("Field_Cellbased3D");
            std::shared_ptr<const Function<Scalar>> pointFunction3D = options.get<Function<Scalar>>("Field_Pointbased3D");


            if(!cellFunction3D)
            {
                infoLog() << "No input field!" << std::endl;
                return;
            }

            std::shared_ptr<const Grid<3>> grid = std::dynamic_pointer_cast< const Grid<3>>(pointFunction3D->domain());

            if(!grid)
            {
                throw std::logic_error( "Wrong type of grid!" );
            }

            const ValueArray<Tensor<double>>& fieldValues = pointFunction3D->values();
            const ValueArray<Point3>& gridPoints = grid->points();

            infoLog() << "Field Values: ";
            for(size_t i = 0; i < gridPoints.size(); ++i)
            {
                infoLog() << fieldValues[i] << std::endl;
            }

            LineSet<3> RidgesAndValleysLines;
            std::list<size_t> RidgesAndValleysIndices;

            setResult("RidgesAndValleys", std::make_shared<LineSet<3>>(RidgesAndValleysLines));
        }
    };

    AlgorithmRegister <RidgesAndValleys> dummy("Hauptaufgabe/RidgesAndValleys", "Visualize Ridges and Valleys.");
}
