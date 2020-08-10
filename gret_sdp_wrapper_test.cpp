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

#include <CGAL/assertions.h>

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
#include <error.h>

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
          std::cout << "Read ply-file: " << file << setw(20) << pc_range[i].size() << std::endl;
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

template <typename PointCloudRange, typename PatchRange, typename TreeRange, typename DistanceRange>
void computeRandomKNNCorrespondences( const PointCloudRange& point_clouds,
                                    PatchRange& patches,
                                    Point_range& merged_point_cloud,
                                    const TreeRange& trees,
                                    const DistanceRange& distances, double max_dist, int num_neighbors){
  for (size_t i = 0; i < merged_point_cloud.size(); i++) {
    for (size_t j = 0; j < patches.size(); j++) {
      Knn search (*trees[j], // using the tree
                    merged_point_cloud[i], // for query point i
                    num_neighbors, // searching for 1 nearest neighbor only
                    0, true, distances[j]); // default parameters
        
        // select random neighbor
        Knn::iterator search_it = search.begin();
        search_it += (num_neighbors == 1) ? 0 : rand()%num_neighbors;

        double dist = search_it->second;
        if(dist < max_dist){
          std::size_t nn
            = search_it->first;
          patches[j].emplace_back(point_clouds[j][nn] , i);
        }
    }
  }                                  
}


double computeLCP(const Point_range& P, Point_range& Q, const double max_dist){
  if(P.size() != Q.size())
    throw runtime_error("P and Q have to be of same size to compute LCP");

  Point_d_map point_d_map = CGAL::make_property_map(Q);
  Kd_tree tree(
    boost::counting_iterator<std::size_t>(0), boost::counting_iterator<std::size_t>(Q.size()),
    Splitter(), Search_traits(point_d_map)
    );
  Distance distance = Distance(point_d_map);
  int common_points = 0;
  for(const auto& point : P){
    Knn search (tree, // using the tree
                point, // for query point i
                1, // searching for 1 nearest neighbor only
                0, true, distance); // default parameters
    double dist = search.begin()->second;
    if(dist < max_dist)
      common_points++;
  }
  return (double) common_points / (double) P.size();
}


int main (int argc, char** argv)
{
  const char* config_fname = (argc>1)?argv[1]:"../../assets/bunny/data/bun.conf";
  const double max_dist_correspondences = (argc>2)? stod(argv[2]):0.00001;
  const double cell_size = (argc>3)? stod(argv[3]):0.01;
  const int num_neighbors = (argc>4)? stod(argv[4]):1;
  const double max_dist_lcp = (argc>5)? stod(argv[5]):0.00001;

  vector<Point_range> point_clouds;
  CGAL::Identity_property_map<Point_3> point_map;
  vector<Kernel::Aff_transformation_3> ground_truth_transformations;

  extractPCAndTrFromStandfordConfFile(config_fname, ground_truth_transformations, point_clouds, point_map);
  int num_point_clouds = point_clouds.size();

  vector<Point_range> transformed_point_clouds(num_point_clouds);
  // construct transformed patches
  for (size_t i = 0; i < num_point_clouds; i++)
    for (size_t j = 0; j < point_clouds[i].size(); j++)
      transformed_point_clouds[i].push_back(point_clouds[i][j].transform(ground_truth_transformations[i].inverse()));

  Point_range merged_point_cloud;
  for (size_t i = 0; i < num_point_clouds; i++)
    for (size_t j = 0; j < transformed_point_clouds[i].size(); j++)
      merged_point_cloud.push_back(transformed_point_clouds[i][j]);

  // sample merged points
  Point_range sampled_point_cloud = merged_point_cloud;
  
  //simplification by clustering using erase-remove idiom
  sampled_point_cloud.erase(CGAL::grid_simplify_point_set(sampled_point_cloud, cell_size),
               sampled_point_cloud.end());
  // Optional: after erase(), use Scott Meyer's "swap trick" to trim excess capacity


  
  //   //parameters
  // cout << "starting wlop" << endl;
  // Point_range sampled_point_range;
  // const int retain_num = 1000;
  // const double retain_percentage = 100 * (double) retain_num / (double) merged_point_range.size();   // percentage of points to retain.
  // const double neighbor_radius = 0.5;   // neighbors size.
  // CGAL::wlop_simplify_and_regularize_point_set<Concurrency_tag>
  //   (merged_point_range, std::back_inserter(sampled_point_range),
  //    CGAL::parameters::select_percentage(retain_percentage)
  //   );
  // cout << "ending wlop" << endl;
  // cout << "sampled size: " << sampled_point_cloud.size() << endl;
  

  int num_global_coordinates = sampled_point_cloud.size();

  cout << "number of global coordinates: " << num_global_coordinates << endl;


  // construct a KD tree in N dimensions 
  std::cout << "Constructing kd-trees" << std::endl;
  std::vector<Kd_tree_sptr> trees(num_point_clouds);
  std::vector<Distance> distances(num_point_clouds);
  for (size_t i = 0; i < num_point_clouds; i++)
    constructKdTree(transformed_point_clouds[i], trees[i], distances[i]);


  // compute correspondences & patches
  vector<vector<Indexed_Point>> patches(num_point_clouds);

  computeRandomKNNCorrespondences(point_clouds, patches, sampled_point_cloud,
                                    trees, distances, max_dist_correspondences, num_neighbors);

  
  CGAL::OpenGR::GRET_SDP<Kernel> matcher;
  matcher.registerPatches(patches, num_global_coordinates, CGAL::parameters::point_map(Point_map())
                                                .vertex_index_map(Index_map()));

  vector<Kernel::Aff_transformation_3> computed_transformations;
  matcher.getTransformations(computed_transformations);

  Point_range registered_points;
  for (size_t i = 0; i < num_point_clouds; i++)
    for (size_t j = 0; j < point_clouds[i].size(); j++)
      registered_points.push_back(point_clouds[i][j].transform(computed_transformations[i]));

  // transfer to other pc frame
  for (size_t i = 0; i < registered_points.size(); i++)
    registered_points[i] = registered_points[i].transform(computed_transformations[0].inverse()).transform(ground_truth_transformations[0]);
  
  double lcp = computeLCP(merged_point_cloud, registered_points, max_dist_lcp);
  cout << "lcp: " << lcp << endl;


  // cgal pcloud to pcl pcloud
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->initCameraParameters();

  PointCloudT::ConstPtr rpcloud = CGAL2PCL_Point_Cloud(registered_points);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> green (rpcloud, 0, 255, 0);
  viewer->addPointCloud<PointNT> (rpcloud, green, "registered point cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "registered point cloud");
  
  PointCloudT::ConstPtr mpcloud = CGAL2PCL_Point_Cloud(merged_point_cloud);
  pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> red (rpcloud, 255, 0, 0);
  viewer->addPointCloud<PointNT> (mpcloud, red, "merged point cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "merged point cloud");

  while (!viewer->wasStopped ())
  {
      viewer->spinOnce (100);
      std::this_thread::sleep_for(100ms);
  }

  return EXIT_SUCCESS;
}

