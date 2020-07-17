#include <CGAL/Simple_cartesian.h>
#include <CGAL/property_map.h>
#include <CGAL/boost/graph/named_params_helper.h>
#include <CGAL/Cartesian_matrix.h>
#include <CGAL/boost/graph/Named_function_parameters.h>
#include <CGAL/OpenGR/gret_sdp.h>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>
#include "boost/tuple/tuple.hpp"

#include <fstream>
#include <iostream>
#include <utility>

#include <Eigen/Dense>


using namespace std;
namespace pt = boost::property_tree;
namespace fs = boost::filesystem;

typedef double Scalar;
typedef CGAL::Simple_cartesian<Scalar> K;
typedef K::Point_3 Point_3;
typedef K::Vector_3 Vector_3;
// indexed point with normal
typedef boost::tuple<Point_3, int, Vector_3> iPwn;
typedef CGAL::Nth_of_tuple_property_map<0, iPwn> iPoint_map;
typedef CGAL::Nth_of_tuple_property_map<1, iPwn> iIndex_map;
typedef CGAL::Nth_of_tuple_property_map<2, iPwn> iNormal_map;
// point with normal
typedef std::pair<Point_3, Vector_3> Pwn;
typedef CGAL::First_of_pair_property_map<Pwn> Point_map;
typedef CGAL::Second_of_pair_property_map<Pwn> Normal_map;

typedef K::Aff_transformation_3 TrafoType;

enum {Dim = 3};
typedef Eigen::Matrix<Scalar, Dim+1, Dim+1> MatrixType;

namespace params = CGAL::parameters;

struct RegistrationProblem {
    int n;
    int m;
    int d;
    std::vector<std::vector<iPwn>> patches;
};

template <typename TrRange>
void extractPatchesAndTrFromConfigFile(const string& configFilePath,  RegistrationProblem& problem, TrRange& transformations);

int main(int argc, const char** argv)
{
    if(argc != 2){
        std::cout << "execute program using: " << "./gret-sdp_with_OpenGR" << " <config/file/path>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string config_file(argv[1]);

    RegistrationProblem problem;
    std::vector<MatrixType> gt_transformations;

    extractPatchesAndTrFromConfigFile(config_file, problem, gt_transformations);

    const int d = problem.d;
    const int n = problem.n;
    const int m = problem.m;
    const vector<std::vector<iPwn>>& patches = problem.patches;

    CGAL::OpenGR::GRET_SDP<K> matcher;
    matcher.registerPatches(patches, n, params::point_map(iPoint_map())
                                                .normal_map(iNormal_map())
                                                .vertex_index_map(iIndex_map()));

    std::vector<TrafoType> transformations;
    matcher.getTransformations(transformations);
    std::vector<Pwn> registered_patches;
    matcher.getRegisteredPatches(registered_patches, params::point_map(Point_map())
                                                .normal_map(Normal_map()));
    
}

template <typename PatchRange>
void readPatch(const string& file, PatchRange& patch){
    int num_points;
    fs::ifstream file_stream(file);
    file_stream >> num_points;
    patch.reserve(num_points);

    Scalar x, y, z;
    int index;
    while(file_stream >> index){
        file_stream >> x;
        file_stream >> y;
        file_stream >> z;        
        patch.emplace_back(Point_3(x,y,z), index, Vector_3());
  }
}

void readTransformation(const string& filePath, MatrixType& trafo){
    int rows, cols;
    fs::ifstream file(filePath);
    if(file.is_open()){
        file >> rows >> cols;
        if(cols != Dim+1 || rows != Dim+1)
            throw std::runtime_error("matrices have to be of size " + to_string(Dim+1) + "x" + to_string(Dim+1));
        for(int i = 0; i < cols; i++)
            for (int j = 0; j < rows; j++)
                file >> trafo(i, j);
    }
}

template <typename TrRange>
void extractPatchesAndTrFromConfigFile(const string& configFilePath,  RegistrationProblem& problem, TrRange& transformations){
    const string workingDir = fs::path(configFilePath).parent_path().native();

    pt::ptree root;
    pt::read_json(configFilePath, root);

    int n = root.get<int>("n");
    int m = root.get<int>("m");
    int d = root.get<int>("d");

    problem.n = n;
    problem.m = m;
    problem.d = d;

    vector< string  > patchFiles;

    for (pt::ptree::value_type &item : root.get_child("patches"))
    {
        patchFiles.push_back(item.second.data());
    }

    if(patchFiles.size() != m)
        throw runtime_error("Number of patches m and number of given patch files is not the same.");

    if(d != Dim)
        throw runtime_error("Dimension of point type has to be " + to_string(Dim));

    // read patch files
    problem.patches.resize(m);
    ifstream patch_file;
    for(int i = 0; i < m; i++){
        readPatch(workingDir + "/" + patchFiles[i], problem.patches[i]);
    }
    
    vector< string  > transformationFiles;
    for (pt::ptree::value_type &item : root.get_child("gt_trafos"))
        transformationFiles.push_back(item.second.data());

    if(transformationFiles.size() != m)
        throw runtime_error("Number of transformations and number of given transformation files is not the same.");

    transformations.reserve(m);
    MatrixType trafo;
    for(int i = 0; i < m; i++){
        readTransformation(workingDir + "/" + transformationFiles[i], trafo);
        transformations.emplace_back(trafo);
    }

}

