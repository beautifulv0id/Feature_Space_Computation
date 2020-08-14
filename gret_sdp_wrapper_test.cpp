#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Classification/Point_set_neighborhood.h>

#include <CGAL/boost/graph/Named_function_parameters.h>
#include <CGAL/boost/graph/named_params_helper.h>

#include <CGAL/Epick_d.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_d.h>
#include <CGAL/Fuzzy_sphere.h>
#include <CGAL/Fuzzy_iso_box.h>

#include <CGAL/IO/read_ply_points.h>
#include <CGAL/IO/write_ply_points.h>

#include <CGAL/grid_simplify_point_set.h>
#include <CGAL/hierarchy_simplify_point_set.h>

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

#include "utils.hpp"

namespace pt = boost::property_tree;
namespace fs = boost::filesystem;

// For computations 3D space
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = Kernel::Point_3;
using Vector_3 = Kernel::Vector_3;
using Point_range = std::vector<Point_3>;
// point with normal
using Indexed_Point = std::pair<Point_3, int>;
using Point_map = CGAL::First_of_pair_property_map<Indexed_Point>;
using Index_map = CGAL::Second_of_pair_property_map<Indexed_Point>;

// KD tree in N dimension
using Search_traits_base = CGAL::Search_traits_3<Kernel>;
using Point_3_map = typename CGAL::Pointer_property_map<Point_3>::type;
using Search_traits = CGAL::Search_traits_adapter<std::size_t, Point_3_map, Search_traits_base>;
using Knn = CGAL::Orthogonal_k_neighbor_search<Search_traits>;
using Tree = typename Knn::Tree;
using Splitter = typename Knn::Splitter;
using Distance = typename Knn::Distance;
using Tree_ptr = std::unique_ptr<Tree>;
using Fuzzy_sphere = CGAL::Fuzzy_sphere<Search_traits>;

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

void constructKdTree(Point_range& feature_range, Tree_ptr& tree, Distance& distance){
  Point_3_map point_d_map = CGAL::make_property_map(feature_range);
  tree.reset(
    new Tree(
    boost::counting_iterator<std::size_t>(0), boost::counting_iterator<std::size_t>(feature_range.size()),
    Splitter(), Search_traits(point_d_map)
    ));
  tree->build();
  distance = Distance(point_d_map);
}


template <typename PointCloudRange, typename PatchRange, typename TreeRange, typename DistanceRange>
void computeRandomKNNCorrespondences( const PointCloudRange& point_clouds,
                                    PatchRange& patches,
                                    Point_range& merged_point_cloud,
                                    const TreeRange& trees,
                                    const DistanceRange& distances, double max_dist){
  for (size_t i = 0; i < merged_point_cloud.size(); i++) {
    Point_3& query = merged_point_cloud[i];
    for (size_t j = 0; j < patches.size(); j++) {
      Fuzzy_sphere fs(query, max_dist, 0, trees[j]->traits());
      if(boost::optional<long unsigned int> nn = trees[j]->search_any_point(fs)){
        patches[j].emplace_back(point_clouds[j][*nn] , i);
      } 
    }
  }        
}


double computeLCP(const Point_range& P, Point_range& Q, const double max_dist){
  if(P.size() != Q.size())
    throw runtime_error("P and Q have to be of same size to compute LCP");

  Point_3_map point_d_map = CGAL::make_property_map(Q);
  Tree tree(
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
  const int retain_num = (argc>2)? stoi(argv[2]):200;
  const double max_dist_correspondences = (argc>3)? stod(argv[3]):0.01;
  //const double max_dist_lcp = (argc>4)? stod(argv[4]):0.001;

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
  sampled_point_cloud.erase(CGAL::hierarchy_simplify_point_set(sampled_point_cloud, 
                            CGAL::parameters::size((double)merged_point_cloud.size()/(double)retain_num)
                            .maximum_variation(0.33)),
                            sampled_point_cloud.end());

  int num_global_coordinates = sampled_point_cloud.size();

  // construct a KD tree in N dimensions 
  std::vector<Tree_ptr> trees(num_point_clouds);
  std::vector<Distance> distances(num_point_clouds);
  for (size_t i = 0; i < num_point_clouds; i++)
    constructKdTree(transformed_point_clouds[i], trees[i], distances[i]);

  // compute correspondences & patches
  vector<vector<Indexed_Point>> patches(num_point_clouds);
  computeRandomKNNCorrespondences(point_clouds, patches, sampled_point_cloud,
                                    trees, distances, max_dist_correspondences);

  
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
  
//  double lcp = computeLCP(merged_point_cloud, registered_points, max_dist_lcp);

  // compute average distance
  long double average_distance = 0;
  for (size_t i = 0; i < merged_point_cloud.size(); i++)
    average_distance += sqrt(CGAL::squared_distance(merged_point_cloud[i], registered_points[i]));
  average_distance = average_distance / (double) merged_point_cloud.size();

  cout << 
    setw(20) << "global coordinates" << 
    setw(24) << "correspond. max dist." <<
//    setw(16) << "lcp max dist." << 
//    setw(11) << "lcp" <<
    setw(16) << "average dist." <<
  endl;

  cout << 
    setw(20) << num_global_coordinates << 
    setw(24) << max_dist_correspondences <<
//    setw(16) << max_dist_lcp << 
//    setw(11) << lcp <<
    setw(16) << average_distance <<
  endl;


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

