#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Classification/Local_eigen_analysis.h>
#include <CGAL/Classification/Point_set_neighborhood.h>

#include <CGAL/Epick_d.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_d.h>

#include <CGAL/IO/read_ply_points.h>
#include <CGAL/IO/write_ply_points.h>

#include <fstream>
#include <iostream>
#include <utility>
#include <vector>
#include <string>


#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/filesystem.hpp>
#include "boost/tuple/tuple.hpp"

namespace pt = boost::property_tree;
namespace fs = boost::filesystem;

// For computations 3D space
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = Kernel::Point_3;
using Point_range = std::vector<Point_3>;
using Neighborhood
= CGAL::Classification::Point_set_neighborhood
  <Kernel, Point_range, CGAL::Identity_property_map<Point_3> >;

using Eigen_analysis = CGAL::Classification::Local_eigen_analysis;

// For computations in feature space
constexpr unsigned int feature_space_dimension = 3;
using Dimension = CGAL::Dimension_tag<feature_space_dimension>;
using Kernel_d = CGAL::Epick_d<Dimension>;
using Point_d = Kernel_d::Point_d;
using Feature_range = std::vector<Point_d>;

// KD tree in N dimension
using Search_traits_base = CGAL::Search_traits_d<Kernel_d, Dimension>;
using Point_d_map = typename CGAL::Pointer_property_map<Point_d>::type;
using Search_traits = CGAL::Search_traits_adapter<std::size_t, Point_d_map, Search_traits_base>;
using Knn = CGAL::Orthogonal_k_neighbor_search<Search_traits>;
using Kd_tree = typename Knn::Tree;
using Splitter = typename Knn::Splitter;
using Distance = typename Knn::Distance;
using Kd_tree_sptr = std::unique_ptr<Kd_tree>;

template <typename PCRange, typename PointMap>
void readPLYsFromConfigFile(const std::string& configFilePath, PCRange& pc_range, const PointMap& point_map){
    const std::string workingDir = fs::path(configFilePath).parent_path().native();

    pt::ptree root;
    pt::read_json(configFilePath, root);

    int m = root.get<int>("m");
    int d = root.get<int>("d");


    std::vector< std::string  > pcs_fnames;

    for (pt::ptree::value_type &item : root.get_child("pcs"))
    {
        pcs_fnames.push_back(item.second.data());
    }

    if(pcs_fnames.size() != m)
        throw std::runtime_error("Number of patches m and number of given patch files is not the same.");


    // read patch files
    pc_range.resize(m);
    std::ifstream pc_file;
    for(int i = 0; i < m; i++){
        pc_file.open(workingDir + "/" + pcs_fnames[i]);
        if(!pc_file ||
            !CGAL::read_ply_points(pc_file, std::back_inserter(pc_range[i]), point_map))
        {
          std::cerr << "Error: cannot read file " << workingDir + "/" + pcs_fnames[i] << std::endl;
          throw std::exception();
        }
        pc_file.close();
    }
}


int main (int argc, char** argv)
{
  const char* config_fname = (argc>1)?argv[1]:"gret-sdp-data/spiral_config.json";

  constexpr unsigned int nb_neighbors = 6;

  std::vector<Point_range> point_ranges;
  CGAL::Identity_property_map<Point_3> point_map;

  readPLYsFromConfigFile(config_fname, point_ranges, point_map);
  int num_point_clouds = point_ranges.size();

  // This creates a 3D neighborhood + computes eigenvalues
  std::vector<Feature_range> feature_ranges;
  feature_ranges.reserve(num_point_clouds);
  for (int i = 0; i < num_point_clouds; ++i)
  {
    const Point_range& point_range = point_ranges[i];
    feature_ranges.push_back(Feature_range());

    Neighborhood neighborhood = Neighborhood(point_range, point_map);
    Eigen_analysis eigen = Eigen_analysis::create_from_point_set
      (point_range, point_map, neighborhood.k_neighbor_query(nb_neighbors));
    // If you want to use a fixed radius instead of KNN, use:
    // Eigen_analysis eigen = Eigen_analysis::create_from_point_set
    //   (points, point_map, neighborhood.sphere_neighbor_query(radius));
    // TODO 2. Populate features with the eigenvalues for each point
    
    Eigen_analysis::Eigenvalues eigenval;
    feature_ranges[i].reserve(point_range.size());
    for (std::size_t j = 0; j < point_range.size(); ++ j)
    {
      eigenval = eigen.eigenvalue(j);
      feature_ranges[i].emplace_back(eigenval[0], eigenval[1], eigenval[2]);
    }
  }
  // This constructs a KD tree in N dimensions (here it's 3 but it
  // could be any dimension). Here, I'm using a trick: the KD tree
  // stores the indices of the points instead of the points themselves
  // (counting iterator just creates a range (0, 1, 2, ..., n)). This
  // allows, when querying the KD tree, to get the *index* of the
  // closest point (instead of the coordinates of the point itself,
  // which would be useless for you).
  std::vector<Kd_tree_sptr> tree_range;
  std::vector<Distance> distances(num_point_clouds);
  for (size_t i = 0; i < num_point_clouds; i++)
  {
    Point_d_map point_d_map = CGAL::make_property_map(feature_ranges[i]);

    Kd_tree kd_tree* = new Kd_tree(
      boost::counting_iterator<std::size_t>(0), boost::counting_iterator<std::size_t>(feature_ranges[i].size()),
      Splitter(), Search_traits(point_d_map)
    );

    tree_range.emplace_back( kd_tree );
    distances.emplace_back(point_d_map)
    kd_tree->build();
  }
  
  
  




  
  // Point_d_map point_d_map = CGAL::make_property_map(features);
  // Kd_tree tree (boost::counting_iterator<std::size_t>(0),
  //               boost::counting_iterator<std::size_t>(features.size()),
  //               Splitter(),
  //               Search_traits(point_d_map));
  // Distance dist (point_d_map);
  // tree.build();

  // std::vector<std::size_t> closest_point_in_feature_space (point_ranges[0].size());

  // // TODO 3. Find closest points in feature spaces (I imagine you
  // // should have one KD Tree per point cloud)
  // for (std::size_t i = 0; i < point_ranges[0].size(); ++ i)
  // {
  //   const Point_d& features_of_point_i = features[i];

  //   // Do the nearest neighbor query
  //   Knn knn (tree, // using the tree
  //            features_of_point_i, // for query point i
  //            1, // searching for 1 nearest neighbor only
  //            0, true, dist); // default parameters

  //   // You get the index of the nearest point. If you constructed the
  //   // KD tree directly on the feature point (without counting
  //   // iterator), you would have a Point_d type returned.
  //   std::size_t index_of_nearest_neighbor
  //     = knn.begin()->first;
  //   std::cout << "query index: " << i << " nn-index: " << index_of_nearest_neighbor << std::endl;
  // }

  // // TODO 4. Call CGAL wrapper with feature


  return EXIT_SUCCESS;
}
