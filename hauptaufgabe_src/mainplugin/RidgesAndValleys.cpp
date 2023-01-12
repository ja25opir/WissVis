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

        std::vector<Point2> calcGradients(const ValueArray<Point2>& gridPoints, const ValueArray<Tensor<double>>& fieldValues)
        {
            double eta = 0.5;
            double gradient1;
            double gradient2;
            std::vector<Point2> interesting_points;

            for(size_t i = 0; i < gridPoints.size()-1-eta; ++i)
            {
                gradient1 = ((fieldValues[i][0] + eta) - fieldValues[i][0]) / eta;
                gradient2 = ((fieldValues[i+1][0] + eta) - fieldValues[i+1][0]) / eta;

                infoLog() << gradient1 << "   " << gradient2 << std::endl;

                if(signbit(gradient1) != signbit(gradient2))
                {
                    infoLog() << "found interesting point" << std::endl;
                    interesting_points.push_back(gridPoints[i]);
                }
            }
            return interesting_points;
        }

        virtual void execute( const Algorithm::Options& options, const volatile bool& /*abortFlag*/ ) override
        {
            std::shared_ptr<const Function<Scalar>> cFunction2D = options.get<Function<Scalar>>("Field_Cellbased2D");
            std::shared_ptr<const Function<Scalar>> pFunction2D = options.get<Function<Scalar>>("Field_Pointbased2D");

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
                const ValueArray<Tensor<double>>& cFieldValues2D = cFunction2D->values();
                const ValueArray<Point2>& cGridPoints2D = cGrid2D->points();
            }

            if(pFunction2D)
            {
                std::shared_ptr<const Grid<2>> pGrid2D = std::dynamic_pointer_cast< const Grid<2>>(pFunction2D->domain());
                const ValueArray<Tensor<double>>& pFieldValues2D = pFunction2D->values();
                const ValueArray<Point2>& pGridPoints2D = pGrid2D->points();

                std::vector<Point2> interesting_points = calcGradients(pGridPoints2D, pFieldValues2D);

                for(size_t i = 0; i < interesting_points.size(); ++i)
                {
                    infoLog() << interesting_points[i] << std::endl;
                }

                setResult("RidgesAndValleys 2D", std::shared_ptr<const Grid<2>>(pGrid2D));
            }

            if(cFunction3D)
            {
                std::shared_ptr<const Grid<3>> cGrid3D = std::dynamic_pointer_cast< const Grid<3>>(cFunction3D->domain());
                const ValueArray<Tensor<double>>& cFieldValues3D = cFunction3D->values();
                const ValueArray<Point3>& cGridPoints3D = cGrid3D->points();
            }

            if(pFunction3D)
            {
                std::shared_ptr<const Grid<3>> pGrid3D = std::dynamic_pointer_cast< const Grid<3>>(pFunction3D->domain());
                const ValueArray<Tensor<double>>& pFieldValues3D = pFunction3D->values();
                const ValueArray<Point3>& pGridPoints3D = pGrid3D->points();

                setResult("RidgesAndValleys 3D", std::shared_ptr<const Grid<3>>(pGrid3D));

            }

        }
    };

    AlgorithmRegister <RidgesAndValleys> dummy("Hauptaufgabe/RidgesAndValleys", "Visualize Ridges and Valleys.");
}
