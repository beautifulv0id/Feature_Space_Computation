#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Classification/Local_eigen_analysis.h>
#include <CGAL/Classification/Point_set_neighborhood.h>

#include <CGAL/boost/graph/Named_function_parameters.h>
#include <CGAL/boost/graph/named_params_helper.h>

#include <CGAL/Epick_d.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_d.h>

#include <CGAL/IO/read_ply_points.h>
#include <CGAL/IO/write_ply_points.h>

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

#include <unordered_map>

namespace pt = boost::property_tree;
namespace fs = boost::filesystem;

// For computations 3D space
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = Kernel::Point_3;
using Point_range = std::vector<Point_3>;
using Neighborhood
= CGAL::Classification::Point_set_neighborhood
  <Kernel, Point_range, CGAL::Identity_property_map<Point_3> >;
// point with normal
using Indexed_Point = std::pair<Point_3, int>;
using Point_map = CGAL::First_of_pair_property_map<Indexed_Point>;
using Index_map = CGAL::Second_of_pair_property_map<Indexed_Point>;

using Eigen_analysis = CGAL::Classification::Local_eigen_analysis;

// For computations in feature space
constexpr unsigned int nb_neighbors = 8;
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
using Kd_tree_sptr = std::shared_ptr<Kd_tree>;

const Knn::FT average_feature_diff = 0.000301027;
const Knn::FT max_feature_diff = average_feature_diff * 1.5;

// matrix graph
using MatrixX = Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;


// A hash function used to hash a pair of any kind 
struct hash_pair { 
    template <class T1, class T2> 
    size_t operator()(const std::pair<T1, T2>& p) const
    { 
        auto hash1 = std::hash<T1>{}(p.first); 
        auto hash2 = std::hash<T2>{}(p.second); 
        return hash1 ^ hash2; 
    } 
};

// map
// key: (pc_idx, curr_global_idx)
// val: (other_global_idx, dist)
using Map = std::unordered_map<std::pair<uint, uint>, std::pair<uint, double>, hash_pair>;
using Map_iterator = typename Map::iterator;
using Map_value_type = typename Map::value_type::second_type;


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
            !CGAL::read_ply_points(pc_file, std::back_inserter(pc_range[i]), CGAL::parameters::point_map(point_map)))
        {
          std::cerr << "Error: cannot read file " << workingDir + "/" + pcs_fnames[i] << std::endl;
          throw std::exception();
        } else {
          std::cout << "Read ply-file: " << workingDir + "/" + pcs_fnames[i] << "\t" << pc_range[i].size() << std::endl;
        }
        pc_file.close();
    }
}


void constructKdTree(Feature_range& feature_range, Kd_tree_sptr& tree, Distance& distance){
  Point_d_map point_d_map = CGAL::make_property_map(feature_range);
  tree.reset(
    new Kd_tree(
    boost::counting_iterator<std::size_t>(0), boost::counting_iterator<std::size_t>(feature_range.size()),
    Splitter(), Search_traits(point_d_map)
    ));
  tree->build();
  distance = Distance(point_d_map);
}

template <typename FeatureRange, typename PointRange, typename PointMap>
void computeFeatureRange(FeatureRange& feature_range, const PointRange& point_range, const PointMap& point_map){
  Neighborhood neighborhood = Neighborhood(point_range, point_map);
  Eigen_analysis eigen = Eigen_analysis::create_from_point_set
    (point_range, point_map, neighborhood.k_neighbor_query(nb_neighbors));
  
  Eigen_analysis::Eigenvalues eigenval;
  feature_range.reserve(point_range.size());
  for (std::size_t j = 0; j < point_range.size(); ++ j)
  {
    eigenval = eigen.eigenvalue(j);
    feature_range.emplace_back(eigenval[0], eigenval[1], eigenval[2]);
  }
}

void removeCorrespondence(Eigen::Ref<MatrixX> feature_graph, int x, int y){
  feature_graph(x,y) -= 1;
  feature_graph(y,x) -= 1;
  feature_graph(x,x) -= 1;
  feature_graph(y,y) -= 1;
}

void addCorrespondence(Eigen::Ref<MatrixX> feature_graph, int x, int y){
  feature_graph(x,y) += 1;
  feature_graph(y,x) += 1;
  feature_graph(x,x) += 1;
  feature_graph(y,y) += 1;
}

int main (int argc, char** argv)
{
  const char* config_fname = (argc>1)?argv[1]:"gret-sdp-data/small_spiral_config.json";

  std::vector<Point_range> point_ranges;
  CGAL::Identity_property_map<Point_3> point_map;

  readPLYsFromConfigFile(config_fname, point_ranges, point_map);
  int num_point_clouds = point_ranges.size();


  // This creates a 3D neighborhood + computes eigenvalues
  std::cout << "Computing feature ranges" << std::endl;
  std::vector<Feature_range> feature_ranges(num_point_clouds);
  for (int i = 0; i < num_point_clouds; ++i)
    computeFeatureRange(feature_ranges[i], point_ranges[i], point_map);
  

  // This constructs a KD tree in N dimensions 
  std::cout << "Constructing kd-trees" << std::endl;
  std::vector<Kd_tree_sptr> tree_range(num_point_clouds);
  std::vector<Distance> distances(num_point_clouds);
  for (size_t i = 0; i < num_point_clouds; i++)
    constructKdTree(feature_ranges[i], tree_range[i], distances[i]);
  

  // compute correspondences
  std::cout << "Computing correspondences" << std::endl;
  uint sum_of_point_cloud_sizes = 0;
  std::vector<uint> pc_start_index;
  pc_start_index.reserve(num_point_clouds);
  for (const auto& point_range : point_ranges){
    pc_start_index.push_back(sum_of_point_cloud_sizes);
    sum_of_point_cloud_sizes += point_range.size();
  }
  // feature graph
  MatrixX feature_graph = MatrixX::Zero(sum_of_point_cloud_sizes, sum_of_point_cloud_sizes);

  // hash maps
  std::vector<Map> map_range(num_point_clouds);
  
  // comute correspondences
  // for every point cloud
  for (size_t current_pc_index = 0; current_pc_index < num_point_clouds; current_pc_index++)
  {
    Kd_tree& current_pc_tree = *tree_range[current_pc_index]; 
    Feature_range& current_pc_features = feature_ranges[current_pc_index];
    Map& current_pc_map = map_range[current_pc_index];
    Map_iterator current_pc_map_it;
    uint current_pc_start_index = pc_start_index[current_pc_index];


    // compare to every other point cloud k
    for (size_t other_pc_index = 0; other_pc_index < num_point_clouds; other_pc_index++)
    {
      if(other_pc_index != current_pc_index){
        Kd_tree& other_pc_tree = *tree_range[other_pc_index]; 
        Feature_range& other_pc_features = feature_ranges[other_pc_index];
        Map& other_pc_map = map_range[other_pc_index];
        Map_iterator other_pc_map_it;
        uint other_pc_start_index = pc_start_index[other_pc_index];


        // every point in current point cloud
        for (size_t current_pc_point_index = 0; current_pc_point_index < point_ranges[current_pc_index].size(); current_pc_point_index++)
        {
            // used to look if point has been added already
            current_pc_map_it = current_pc_map.find(std::make_pair(other_pc_index, current_pc_point_index));

            // only if point wasn't matched before
            if(current_pc_map_it == current_pc_map.end()){

              const Point_d& current_pc_point_features = current_pc_features[current_pc_point_index];
              // Do the nearest neighbor query
              Knn other_pc_knn (other_pc_tree, // using the tree
                      current_pc_point_features, // for query point i
                      1, // searching for 1 nearest neighbor only
                      0, true, distances[other_pc_index]); // default parameters

              // index of nearest neighbor
              std::size_t other_pc_nn_index
                = other_pc_knn.begin()->first;
              // distance between current point and nn in other point cloud
              double current_dist = other_pc_knn.begin()->second;

              // search other point in map
              other_pc_map_it = other_pc_map.find(std::make_pair(current_pc_index, other_pc_nn_index));

              // if nn wasn't matches by point within same point cloud before
              if(other_pc_map_it == other_pc_map.end()){
                // match back
                const Point_d& nearest_neighbor_features = other_pc_features[other_pc_nn_index];
                Knn current_knn (current_pc_tree, // using the tree
                    nearest_neighbor_features, // for query point i
                    1, // searching for 1 nearest neighbor only
                    0, true, distances[current_pc_index]); // default parameters

                // index of nearest neighbor
                std::size_t current_pc_nn_index
                  = current_knn.begin()->first;

                // if correspondence matches back add correspondence
                if(current_pc_nn_index == (current_pc_index + current_pc_point_index)){
                  // update graph
                  addCorrespondence( 
                    feature_graph,
                    other_pc_start_index + other_pc_nn_index, 
                    current_pc_start_index + current_pc_point_index
                  );

                  // update maps
                  other_pc_map[std::make_pair(current_pc_index, other_pc_nn_index)]
                  = std::make_pair(current_pc_start_index + current_pc_point_index, current_dist);
                  
                  current_pc_map[std::make_pair(other_pc_index, current_pc_point_index)] 
                    = std::make_pair(other_pc_start_index + other_pc_nn_index, current_dist);
                }
              } 
            }
        }
      }
    }
  }

  // CALCULATE number of global coordinates
  // number of global cooridinates
  uint global_coordinate = 0;
  int correspondences;
  std::vector<std::vector<Indexed_Point>> patches(num_point_clouds);
  int current_pc_point_global_index;
  for (size_t current_pc_index = 0; current_pc_index < num_point_clouds; current_pc_index++)
  {
    std::vector<Indexed_Point>& current_patch = patches[current_pc_index];
    Point_range& current_point_range = point_ranges[current_pc_index];
    Map& current_pc_map = map_range[current_pc_index];
    Map_iterator current_pc_map_it;
    uint current_pc_start_index = pc_start_index[current_pc_index];
    current_pc_point_global_index = pc_start_index[current_pc_index];

    // for all points within current point cloud
    for (size_t current_pc_point_index = 0; current_pc_point_index < point_ranges[current_pc_index].size(); 
                current_pc_point_index++, current_pc_point_global_index++)
    {
      correspondences = feature_graph(current_pc_point_global_index, current_pc_point_global_index);
      if (correspondences > 0){
        // push back point
        current_patch.push_back(std::make_pair(current_point_range[current_pc_point_index], global_coordinate));


        // add other correspondences and update graph
        for (size_t other_pc_index = 0; other_pc_index < num_point_clouds; other_pc_index++)
        {
          // search for correspondences
          if(other_pc_index != current_pc_index){

            current_pc_map_it = current_pc_map.find(std::make_pair(other_pc_index, current_pc_point_index));

            // if correspondence was found add points to patches and update feature_graph
            if(current_pc_map_it != current_pc_map.end()){
              Point_range& other_point_range = point_ranges[other_pc_index];
              std::vector<Indexed_Point>& other_patch = patches[other_pc_index];
              uint other_pc_start_index = pc_start_index[other_pc_index];
              uint other_pc_point_global_index = get<uint>(current_pc_map_it->second);
              uint other_pc_point_index = other_pc_point_global_index - other_pc_start_index;
              other_patch.push_back(std::make_pair(other_point_range[other_pc_point_index], global_coordinate));

              removeCorrespondence(
                feature_graph,
                current_pc_point_global_index,
                other_pc_point_global_index
              );

              // remove from maps
              Map& other_pc_map = map_range[other_pc_index];
              current_pc_map.erase(current_pc_map_it);
              other_pc_map.erase(std::make_pair(current_pc_index, other_pc_point_index));
            }
          }
        }

        // increase number of global coorinates
        global_coordinate++;
      }
    }
  }
  

  int num_global_coordinates = global_coordinate - 1;
  
  std::cout << "num_global_coordinates: " << num_global_coordinates << std::endl;
  
  // // TODO 4. Call CGAL wrapper with feature
  CGAL::OpenGR::GRET_SDP<Kernel> matcher;
  matcher.registerPatches(patches, num_global_coordinates, CGAL::parameters::point_map(Point_map())
                                                .vertex_index_map(Index_map()));

  std::vector<Indexed_Point> registered_patches;
  matcher.getRegisteredPatches(registered_patches, CGAL::parameters::point_map(Point_map()));

  std::ofstream out("registered_point_clouds.ply");
    if (!out ||
      !CGAL::write_ply_points(
        out, registered_patches,
        CGAL::parameters::point_map(Point_map())))
  {
    return EXIT_FAILURE;
  }

    std::cout << "Registered point clouds have bin written to: "
            << "registered_point_clouds.ply.\n";

  return EXIT_SUCCESS;
}
