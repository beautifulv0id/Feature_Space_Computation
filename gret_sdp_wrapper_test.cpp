#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Classification/Point_set_neighborhood.h>

#include <CGAL/boost/graph/Named_function_parameters.h>
#include <CGAL/boost/graph/named_params_helper.h>

#include <CGAL/Epick_d.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_d.h>

#include <CGAL/IO/read_ply_points.h>
#include <CGAL/IO/write_ply_points.h>

#include <CGAL/grid_simplify_point_set.h>
#include <CGAL/wlop_simplify_and_regularize_point_set.h>

// Concurrency
#ifdef CGAL_LINKED_WITH_TBB
typedef CGAL::Parallel_tag Concurrency_tag;
#else
typedef CGAL::Sequential_tag Concurrency_tag;
#endif


#define CGAL_LINKED_WITH_OPENGR
#include <CGAL/OpenGR/gret_sdp.h>

#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
#include <string>

#include <Eigen/Dense>

#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>
#include "boost/tuple/tuple.hpp"

#include <thread>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>

namespace pt = boost::property_tree;
namespace fs = boost::filesystem;

// For computations 3D space
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = Kernel::Point_3;
using Vector_3 = Kernel::Vector_3;
using Point_range = std::vector<Point_3>;
using Neighborhood
= CGAL::Classification::Point_set_neighborhood
  <Kernel, Point_range, CGAL::Identity_property_map<Point_3> >;
// point with normal
using Indexed_Point = std::pair<Point_3, int>;
using Point_map = CGAL::First_of_pair_property_map<Indexed_Point>;
using Index_map = CGAL::Second_of_pair_property_map<Indexed_Point>;


// For computations in feature space
const unsigned int Neighbors = 1;
const double delta = 0.00001;
constexpr unsigned int feature_space_dimension = 3;
using Dimension = CGAL::Dimension_tag<feature_space_dimension>;
using Kernel_d = CGAL::Epick_d<Dimension>;
using Point_d = Kernel_d::Point_d;
using Feature_range = std::vector<Point_d>;

// KD tree in N dimension
using Search_traits_base = CGAL::Search_traits_3<Kernel>;
using Point_d_map = typename CGAL::Pointer_property_map<Point_3>::type;
using Search_traits = CGAL::Search_traits_adapter<std::size_t, Point_d_map, Search_traits_base>;
using Knn = CGAL::Orthogonal_k_neighbor_search<Search_traits>;
using Kd_tree = typename Knn::Tree;
using Splitter = typename Knn::Splitter;
using Distance = typename Knn::Distance;
using Kd_tree_sptr = std::unique_ptr<Kd_tree>;

using Scalar = double;
constexpr int Dim = 3;
typedef Eigen::Transform<Scalar, 3, Eigen::Affine> Transform;

typedef pcl::PointXYZ PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;


// pcl for visualization
using namespace std::chrono_literals;
using namespace std;

// matrix graph
using MatrixX = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;


struct Correspondence {
  unsigned int pc_;
  unsigned int idx_;

  Correspondence(int pc, int idx) : pc_(pc), idx_(idx) {}
};


template <typename PointMap>
void extractPCAndTrFromStandfordConfFile(
        const std::string &confFilePath,
        std::vector<Kernel::Aff_transformation_3>& transforms,
        std::vector<Point_range>& pc_range,
        const PointMap& point_map
        ){
    using namespace boost;
    using namespace std;
    vector<string> files;
    
    //VERIFY (filesystem::exists(confFilePath) && filesystem::is_regular_file(confFilePath));

    // extract the working directory for the configuration path
    const string workingDir = filesystem::path(confFilePath).parent_path().native();
    //VERIFY (filesystem::exists(workingDir));

    // read the configuration file and call the matching process
    string line;
    ifstream confFile;
    confFile.open(confFilePath);
    //VERIFY (confFile.is_open());

    while ( getline (confFile,line) )
    {
        istringstream iss (line);
        vector<string> tokens{istream_iterator<string>{iss},
                              istream_iterator<string>{}};

        // here we know that the tokens are:
        // [0]: keyword, must be bmesh
        // [1]: 3D object filename
        // [2-4]: target translation with previous object
        // [5-8]: target quaternion with previous object

        if (tokens.size() == 9){
            if (tokens[0].compare("bmesh") == 0){
                // skip problematic models
                if (tokens[1].compare("ArmadilloSide_165.ply") == 0) continue;

                string inputfile = filesystem::path(confFilePath).parent_path().string()+string("/")+tokens[1];
                //VERIFY(filesystem::exists(inputfile) && filesystem::is_regular_file(inputfile));

                // build the Eigen rotation matrix from the rotation and translation stored in the files
                Eigen::Matrix<Scalar, Dim, 1> tr (
                            std::atof(tokens[2].c_str()),
                            std::atof(tokens[3].c_str()),
                            std::atof(tokens[4].c_str()));

                Eigen::Quaternion<Scalar> quat(
                            std::atof(tokens[8].c_str()), // eigen starts by w
                            std::atof(tokens[5].c_str()),
                            std::atof(tokens[6].c_str()),
                            std::atof(tokens[7].c_str()));

                quat.normalize();

                Transform transform (Transform::Identity());
                transform.rotate(quat);
                transform.translate(-tr);

                transforms.emplace_back(
                transform(0,0), transform(0,1), transform(0,2), transform(0,3),
                transform(1,0), transform(1,1), transform(1,2), transform(1,3),
                transform(2,0), transform(2,1), transform(2,2), transform(2,3)
                );

                files.push_back(inputfile);
            }
        }
    }
    confFile.close();

    std::ifstream pc_file;
    int num_point_clouds = files.size();
    pc_range.resize(num_point_clouds);
    for(int i = 0; i < num_point_clouds; i++){
        const string& file = files[i];
        pc_file.open(file);
        if(!pc_file ||
            !CGAL::read_ply_points(pc_file, std::back_inserter(pc_range[i]), CGAL::parameters::point_map(point_map)))
        {
          std::cerr << "Error: cannot read file " << file << std::endl;
          throw std::exception();
        } else {
          std::cout << "Read ply-file: " << file << "\t" << pc_range[i].size() << std::endl;
        }
        pc_file.close();
    }
}


void constructKdTree(Point_range& feature_range, Kd_tree_sptr& tree, Distance& distance){
  Point_d_map point_d_map = CGAL::make_property_map(feature_range);
  tree.reset(
    new Kd_tree(
    boost::counting_iterator<std::size_t>(0), boost::counting_iterator<std::size_t>(feature_range.size()),
    Splitter(), Search_traits(point_d_map)
    ));
  tree->build();
  distance = Distance(point_d_map);
}


template <typename PointRange>
pcl::PointCloud<pcl::PointXYZ>::ConstPtr CGAL2PCL_Point_Cloud(const PointRange& cgal_pcloud){
    PointCloudT::Ptr pcl_pcloud (new PointCloudT);
    for (auto& point : cgal_pcloud)
    {
        PointNT pcl_point(point.x(), point.y(), point.z());
        pcl_pcloud->push_back(pcl_point);
    }
    return pcl_pcloud;
}

int main (int argc, char** argv)
{
  const char* config_fname = (argc>1)?argv[1]:"../../assets/bunny/data/bun.conf";

  vector<Point_range> point_ranges;
  CGAL::Identity_property_map<Point_3> point_map;
  vector<Kernel::Aff_transformation_3> transform_range;

  extractPCAndTrFromStandfordConfFile(config_fname, transform_range, point_ranges, point_map);
  int num_point_clouds = point_ranges.size();

  vector<Point_range> transformed_point_ranges(num_point_clouds);
  for (size_t i = 0; i < num_point_clouds; i++)
    for (size_t j = 0; j < point_ranges[i].size(); j++)
      transformed_point_ranges[i].push_back(point_ranges[i][j].transform(transform_range[i].inverse()));

  Point_range merged_point_range;
  for (size_t i = 0; i < num_point_clouds; i++)
    for (size_t j = 0; j < transformed_point_ranges[i].size(); j++)
      merged_point_range.push_back(transformed_point_ranges[i][j]);

  //simplification by clustering using erase-remove idiom
  double cell_size = 0.01;
  merged_point_range.erase(CGAL::grid_simplify_point_set(merged_point_range, cell_size),
               merged_point_range.end());
  // Optional: after erase(), use Scott Meyer's "swap trick" to trim excess capacity

  cout << "merged size: " << merged_point_range.size() << endl;

  // sample merged points
  Point_range& sampled_point_range = merged_point_range;

  //   //parameters
  // cout << "starting wlop" << endl;
  // const int retain_num = 1000;
  // const double retain_percentage = 100 * (double) retain_num / (double) merged_point_range.size();   // percentage of points to retain.
  // const double neighbor_radius = 0.5;   // neighbors size.
  // CGAL::wlop_simplify_and_regularize_point_set<Concurrency_tag>
  //   (merged_point_range, std::back_inserter(sampled_point_range),
  //    CGAL::parameters::select_percentage(retain_percentage)
  //    );
  // cout << "ending wlop" << endl;
  // cout << "sampled size: " << sampled_point_range.size() << endl;
  
  int num_global_coordinates = sampled_point_range.size();

  // construct a KD tree in N dimensions 
  std::cout << "Constructing kd-trees" << std::endl;
  std::vector<Kd_tree_sptr> tree_range(num_point_clouds);
  std::vector<Distance> distance_range(num_point_clouds);
  for (size_t i = 0; i < num_point_clouds; i++)
    constructKdTree(transformed_point_ranges[i], tree_range[i], distance_range[i]);


  // compute correspondences & patches
  vector<vector<Indexed_Point>> patch_range(num_point_clouds);
  vector<vector<Correspondence>> correspondences_range;
  for (size_t i = 0; i < sampled_point_range.size(); i++)
  {
    vector<Correspondence> correspondences;
    for (size_t j = 0; j < num_point_clouds; j++)
    {
      Knn search (*tree_range[j], // using the tree
                    sampled_point_range[i], // for query point i
                    Neighbors, // searching for 1 nearest neighbor only
                    0, true, distance_range[j]); // default parameters
       double dist = search.begin()->second;
       if(dist < delta){
        // index of nearest neighbor
        std::size_t nn
          = search.begin()->first;
        correspondences.emplace_back(j, nn);
        patch_range[j].emplace_back(point_ranges[j][nn] , i);
       }
    }
    correspondences_range.push_back(move(correspondences));
  }
  

  CGAL::OpenGR::GRET_SDP<Kernel> matcher;
  matcher.registerPatches(patch_range, num_global_coordinates, CGAL::parameters::point_map(Point_map())
                                                .vertex_index_map(Index_map()));

  vector<Kernel::Aff_transformation_3> trafos;
  matcher.getTransformations(trafos);

  Point_range registered_points;
  for (size_t i = 0; i < num_point_clouds; i++)
    for (size_t j = 0; j < point_ranges[i].size(); j++)
      registered_points.push_back(point_ranges[i][j].transform(trafos[i]));


  // cgal pcloud to pcl pcloud
  PointCloudT::ConstPtr pcloud = CGAL2PCL_Point_Cloud(registered_points);
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<PointNT> (pcloud, "merged point cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "merged point cloud");
  viewer->initCameraParameters();

  while (!viewer->wasStopped ())
  {
      viewer->spinOnce (100);
      std::this_thread::sleep_for(100ms);
  }

  return EXIT_SUCCESS;
}

