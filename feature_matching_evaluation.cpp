#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/boost/graph/Named_function_parameters.h>
#include <CGAL/boost/graph/named_params_helper.h>
#include <CGAL/Classification/Local_eigen_analysis.h>
#include <CGAL/Classification/Point_set_neighborhood.h>
#include <CGAL/Epick_d.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_d.h>


#include "utils.hpp"

// For computations 3D space
using Kernel = CGAL::Exact_predicates_inexact_constructions_kernel;
using Point_3 = Kernel::Point_3;
using Vector_3 = Kernel::Vector_3;
using Point_range = std::vector<Point_3>;

using Eigen_analysis = CGAL::Classification::Local_eigen_analysis;
using Neighborhood
= CGAL::Classification::Point_set_neighborhood
  <Kernel, Point_range, CGAL::Identity_property_map<Point_3> >;

// For computations in feature space
constexpr unsigned int num_features = 5;
constexpr unsigned int nb_neighbors[] = {4, 8, 16, 32, 64};
constexpr unsigned int feature_space_dimension = num_features * 3;
using Dimension = CGAL::Dimension_tag<feature_space_dimension>;
using Kernel_d = CGAL::Epick_d<Dimension>;
using Point_d = Kernel_d::Point_d;
using Feature_range = std::vector<Point_d>;

// KD tree in N dimension
using Search_traits_base = CGAL::Search_traits_d<Kernel_d, Dimension>;
using Point_3_map = typename CGAL::Pointer_property_map<Point_d>::type;
using Search_traits = CGAL::Search_traits_adapter<std::size_t, Point_3_map, Search_traits_base>;
using Knn = CGAL::Orthogonal_k_neighbor_search<Search_traits>;
using Tree = typename Knn::Tree;
using Splitter = typename Knn::Splitter;
using Distance = typename Knn::Distance;
using Tree_ptr = std::unique_ptr<Tree>;

using CorrespondencesQueue = std::priority_queue<Correspondences, std::vector<Correspondences>, Correspondences>;

using namespace std;



template <typename FeatureRange, typename PointRange, typename PointMap>
void computeEigenFeatureRange(FeatureRange& feature_range, const PointRange& point_range, const PointMap& point_map);

void constructKdTree(Feature_range& feature_range, Tree_ptr& tree, Distance& distance);

template <typename PointCloudRange, typename FeatureRanges, typename TreeRange, 
          typename DistancesRange, typename CorrespondenceRange>
void computeCorrespondences(const PointCloudRange& point_clouds, const FeatureRanges& feature_ranges,
                            const TreeRange& trees, const DistancesRange& distances, 
                            CorrespondenceRange& correspondences_queue);

template <typename TreeRange, typename FeatureRanges, typename DistancesRange>
bool matchBack( Correspondences& correspondences, 
                const TreeRange& tree_range,
                const FeatureRanges& feature_ranges,
                const DistancesRange& distance_range,
                const PointIndexPair& current);


template <typename CorrespondenceQueue, typename CorrespondenceRange>
void correspondenceQueue2Range(CorrespondenceQueue& queue, CorrespondenceRange& range, const int N);

template <typename PointCloudRange, typename TransformationRange, typename CorrespondenceRange>
Distances evaluateCorrespondences( const PointCloudRange& point_clouds, const TransformationRange& transformations, 
                                          const CorrespondenceRange& correspondences_range);

template <typename PointCloudRange, typename TransformationRange, typename CorrespondenceRange>
Distances evaluateCorrespondences( const PointCloudRange& point_clouds, const TransformationRange& transformations, 
                                          const CorrespondenceRange& correspondences_range){
  double average_dist = 0;
  double current_dist;
  double min_dist = 999999;
  double max_dist = -1;
  int num_global_coordinates = correspondences_range.size();
  for (int global_coordinate = 0; global_coordinate < num_global_coordinates; global_coordinate++){
    const Correspondences& correspondences = correspondences_range[global_coordinate];
    double correspondence_average_dist = 0;
    int num_correspondences = correspondences.size();
    for (int i = 0; i < num_correspondences; i++){
      const PointIndexPair& current = correspondences[i];
      for (int j = i+1; j < correspondences.size(); j++){
        const PointIndexPair& other = correspondences[j];
        current_dist = CGAL::squared_distance( point_clouds[current.pc_][current.idx_].transform(transformations[current.pc_].inverse()),
                                                        point_clouds[other.pc_][other.idx_].transform(transformations[other.pc_].inverse()));
        correspondence_average_dist += current_dist;
        if (current_dist < min_dist)
          min_dist = current_dist;
        if (current_dist > max_dist)
          max_dist = current_dist;
        
      }
    }
    correspondence_average_dist /= (double) ( num_correspondences * ( num_correspondences - 1 ) / 2);
    average_dist += correspondence_average_dist;
  }
  average_dist /= (double) num_global_coordinates;

  return Distances(min_dist, max_dist, average_dist);
} 

int main (int argc, char** argv)
{
  const char* config_fname = (argc>1)?argv[1]:"../../assets/bunny/data/bun.conf";
  const int N = (argc>2)? stoi(argv[2]):200;

  

  vector<Point_range> point_clouds;
  CGAL::Identity_property_map<Point_3> point_map;
  vector<Kernel::Aff_transformation_3> ground_truth_transformations;

  extractPCAndTrFromStandfordConfFile(config_fname, ground_truth_transformations, point_clouds, point_map);
  int num_point_clouds = point_clouds.size();

  // compute features
  std::vector<Feature_range> feature_ranges(num_point_clouds);
  for (int i = 0; i < num_point_clouds; ++i)
    computeEigenFeatureRange(feature_ranges[i], point_clouds[i], point_map);

  // construct kd trees
  std::vector<Tree_ptr> trees(num_point_clouds);
  std::vector<Distance> distances(num_point_clouds);
  for (size_t i = 0; i < num_point_clouds; i++)
    constructKdTree(feature_ranges[i], trees[i], distances[i]);

  
  // compute correspondences
  CorrespondencesQueue correspondences_queue;
  computeCorrespondences(point_clouds, feature_ranges, trees, distances, correspondences_queue);

  // compute vector of N or less global coordinates
  int num_global_coordinates = correspondences_queue.size() < N ? correspondences_queue.size() : N;
  std::vector<Correspondences> correspondences_range;
  correspondenceQueue2Range(correspondences_queue, correspondences_range, num_global_coordinates);
  
  // evaluate correspondences
  Distances dist = evaluateCorrespondences(point_clouds, ground_truth_transformations, correspondences_range);
  std::cout << "average correspondence distance = " << dist.average_ << std::endl;

  visualizeCorrespondences(point_clouds, ground_truth_transformations, dist, correspondences_range, true, 0.1);
}

template <typename FeatureRange, typename PointRange, typename PointMap>
void computeEigenFeatureRange(FeatureRange& feature_range, const PointRange& point_range, const PointMap& point_map){
  typedef typename Kernel_d::FT Scalar;
  Neighborhood neighborhood = Neighborhood(point_range, point_map);
  Eigen_analysis eigen[num_features];

  for (size_t i = 0; i < num_features; i++)
    eigen[i] = Eigen_analysis::create_from_point_set
      (point_range, point_map, neighborhood.k_neighbor_query(nb_neighbors[i]));
  
  feature_range.reserve(point_range.size());  
  Scalar* feature_arr = (Scalar*) malloc(num_features * 3 * sizeof(Scalar));
  for (std::size_t j = 0; j < point_range.size(); ++ j)
  {
    for (size_t i = 0; i < num_features; i++){
      const Eigen_analysis::Eigenvalues& eigenval = eigen[i].eigenvalue(j);
      feature_arr[i*3]   = eigenval[0];
      feature_arr[i*3+1] = eigenval[1];
      feature_arr[i*3+2] = eigenval[2];
    }
    feature_range.emplace_back(num_features*3, feature_arr, feature_arr+(num_features*3));
  }

  delete feature_arr;
}


void constructKdTree(Feature_range& feature_range, Tree_ptr& tree, Distance& distance){
  Point_3_map point_d_map = CGAL::make_property_map(feature_range);
  tree.reset(
    new Tree(
    boost::counting_iterator<std::size_t>(0), boost::counting_iterator<std::size_t>(feature_range.size()),
    Splitter(), Search_traits(point_d_map)
    ));
  tree->build();
  distance = Distance(point_d_map);
}


// favours correspondences with points from the beginning point clouds
template <typename PointCloudRange, typename FeatureRanges, typename TreeRange, typename DistancesRange, typename CorrespondenceRange>
void computeCorrespondences(
                            const PointCloudRange& point_ranges,
                            const FeatureRanges& feature_ranges,
                            const TreeRange& tree_range,
                            const DistancesRange& distance_range,
                            CorrespondenceRange& correspondences_queue){
  uint num_point_clouds = point_ranges.size();
  // hash maps
  std::vector<Map> map_range(num_point_clouds);
  bool correspondence_valid = false;
  // for every point cloud
  for (size_t current_pc_index = 0; current_pc_index < num_point_clouds; current_pc_index++)
  {
    const Feature_range& current_pc_features = feature_ranges[current_pc_index];
    Map& current_pc_map = map_range[current_pc_index];
    // every point in current point cloud
    for (size_t current_pc_point_index = 0; current_pc_point_index < point_ranges[current_pc_index].size(); current_pc_point_index++)
    {
      correspondence_valid = false;
      Correspondences correspondences;
      correspondences.add(current_pc_index, current_pc_point_index);

      // search nn in every other point cloud
      for (size_t other_pc_index = 0; other_pc_index < num_point_clouds; other_pc_index++)
      {
        if(other_pc_index != current_pc_index){
          // if correspondence does not already exist
          if(!correspondenceExists(current_pc_map, other_pc_index, current_pc_point_index)){
            Tree& other_pc_tree = *tree_range[other_pc_index]; 
            Map& other_pc_map = map_range[other_pc_index];

            const Point_d& current_pc_point_features = current_pc_features[current_pc_point_index];
            Knn knn_in_other_pc (other_pc_tree, current_pc_point_features,
                              1, 0, true, distance_range[other_pc_index]);
            std::size_t nn_in_other_pc = knn_in_other_pc.begin()->first;

            // if nn wasn't matched before
            if(!correspondenceExists(other_pc_map, current_pc_index, nn_in_other_pc)){
              PointIndexPair other(other_pc_index, nn_in_other_pc);
              if(matchBack(correspondences, tree_range, feature_ranges, distance_range, other))
                correspondence_valid = true;
            } 
          }
        }
      }
      // update maps and add correspondence
      if (correspondence_valid){
        updateMaps(correspondences, map_range);
        correspondences_queue.push(correspondences);
      }
    }
  }  
}



// checks if current matches both ways with all previous points in correspondences
// returns true if so and updates correspondences accordingly
template <typename TreeRange, typename FeatureRanges, typename DistancesRange>
bool matchBack( Correspondences& correspondences, 
                const TreeRange& tree_range,
                const FeatureRanges& feature_ranges,
                const DistancesRange& distance_range,
                const PointIndexPair& current){
  const Point_d& current_features = feature_ranges[current.pc_][current.idx_];
  bool matches_back = true;
  double accum_dist = 0;
  for (const PointIndexPair& other : correspondences.vec()){
    Knn knn_in_other (*tree_range[other.pc_], current_features, 
              1, 0, true, distance_range[other.pc_]); 

    std::size_t nn_in_other = knn_in_other.begin()->first;

    // if it matches to the existing correspondences
    // match back from existing to current
    if (nn_in_other == other.idx_){
      const Point_d& other_features = feature_ranges[other.pc_][other.idx_];
      Knn knn_in_current (*tree_range[current.pc_], other_features, 
              1, 0, true, distance_range[current.pc_]); 

      std::size_t nn_in_current = knn_in_current.begin()->first;

      if(nn_in_current == current.idx_)
        accum_dist += knn_in_current.begin()->second;
      else{
        matches_back = false;
        break;
      }
    }
    else {
      matches_back = false;
      break;
    }         
  }

  if(matches_back){
    // add correspondence
    correspondences.add(current);
    correspondences.updateTotalDistance(accum_dist);
    return true;
  }

  return false;
}

template <typename CorrespondenceQueue, typename CorrespondenceRange>
void correspondenceQueue2Range(CorrespondenceQueue& queue, CorrespondenceRange& range, const int N){
  range.clear();
  range.reserve(N);
  for (int i = 0; i < N; i++){
    range.push_back(queue.top());
    queue.pop();
  }
}