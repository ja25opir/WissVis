#include <fantom/algorithm.hpp>
#include <fantom/dataset.hpp>
#include <fantom/graphics.hpp>
#include <fantom/register.hpp>
#include <math.h>
#include <valarray>
#include <map>
#include "helpers.h"
#include <fantom-plugins/utils/Graphics/ObjectRenderer.hpp>

using namespace fantom;

namespace
{
    class RidgesAndValleys : public VisAlgorithm
    {

    public:
        struct Options : public VisAlgorithm::Options
        {
            Options( fantom::Options::Control& control )
                : VisAlgorithm::Options( control )
            {
                add<Field<2,Scalar>>( "Field_Cellbased2D", "A 2D cell based scalar field", definedOn<Grid<2>>(Grid<2>::Cells));
                add<Field<2,Scalar>>( "Field_Pointbased2D", "A 2D point based scalar field", definedOn<Grid<2>>(Grid<2>::Points));

                add<Field<3,Scalar>>( "Field_Cellbased3D", "A 3D cell based scalar field", definedOn<Grid<3>>(Grid<3>::Cells));
                add<Field<3,Scalar>>( "Field_Pointbased3D", "A 3D point basedscalar field", definedOn<Grid<3>>(Grid<3>::Points));

                add<double>("Epsilon", "Epsilon value for gradient calculation", 1e-4);
            }
        };

        /*
        struct DataOutputs : public DataAlgorithm::DataOutputs
        {
            DataOutputs(fantom::DataOutputs::Control& control)
                : DataAlgorithm::DataOutputs(control)
            {
                add <const Grid<2>> ("RidgesAndValleys 2D");
                add <const Grid<3>> ("RidgesAndValleys 3D");
            }
        };*/

        struct VisOutputs : public VisAlgorithm::VisOutputs
        {
            VisOutputs( fantom::VisOutputs::Control& control )
                : VisAlgorithm::VisOutputs( control )
            {
                addGraphics("Markers");
            }
        };

        RidgesAndValleys (InitData& data)
            : VisAlgorithm (data)
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

        std::vector<Point3> isInterestingCell(const ValueArray<Point2>& gridPoints, Cell& cell, const ValueArray<Scalar>& fieldValues, std::shared_ptr<const Field<2, Scalar>> field, double epsilon)
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
                        Point2 edgeCenter2D = getEdgeCenter2D(gridPoints, cell, edges[j]);
                        Point3 edgeCenter3D = {edgeCenter2D[0], edgeCenter2D[1], 0};
                        edgeCenters.push_back(edgeCenter3D);
                    }
                }
            }
            return edgeCenters;
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


        void markPoints(std::vector<Point3>& points, Color& color)
        {
            auto const& system = graphics::GraphicsSystem::instance();
            auto performanceObjectRenderer = std::make_shared<graphics::ObjectRenderer>(system);
            performanceObjectRenderer->reserve(graphics::ObjectRenderer::ObjectType::SPHERE, points.size());

            for(size_t i = 0; i != points.size(); ++i)
            {
                performanceObjectRenderer->addSphere(points[i], 0.35, color);
            }

            setGraphics("Markers", performanceObjectRenderer->commit());
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
                std::map<int, std::vector<Point3>> ridgeValleyMap;
                std::vector<int> maximumCellsIndices;

                std::list<Point3> pointList;


                for(size_t i = 0; i < pGrid2D->numCells(); ++i)
                {
                    Cell cell = pGrid2D->cell(i);
                    std::vector<Point3> edgePoints = isInterestingCell(pGridPoints2D, cell, pFieldValues2D, pField2D, epsilon);

                    for (size_t i = 0; i < edgePoints.size(); ++i) {
                        pointList.push_back(Point3( edgePoints[i] ));
                    }

                    if(!edgePoints.empty())
                    {
                        //infoLog() << "------------------found interesting cell at: " << i << std::endl;
                        interestingCells.push_back(cell);
                        interestingCellsIndices.push_back(i);
                        ridgeValleyMap.insert({i, edgePoints});
                    }
                }

                std::vector<Point3> markerPoints(pointList.begin(), pointList.end());

                Color color( 0.75, 0.75, 0.0 );
                markPoints(markerPoints, color);

                infoLog() << "interesting cells found: ";
                infoLog() << ridgeValleyMap.size() << std::endl;

                //infoLog() << interestingCells.size() << std::endl;

                /*for(size_t j = 0; j < interestingCells.size(); ++j)
                {
                    //infoLog() << "cell indices: " << interestingCellsIndices[j] << std::endl;
                    if(isMaximum(pGridPoints2D, interestingCells[j], pField2D, epsilon))
                    {
                        //infoLog() << "------------------found extrema cell at: " << interestingCellsIndices[j] << std::endl;
                        maximumCellsIndices.push_back(interestingCellsIndices[j]);
                    }
                }

                setResult("RidgesAndValleys 2D", std::shared_ptr<const Grid<2>>(pGrid2D));*/
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
