#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Classification/Local_eigen_analysis.h>
#include <CGAL/Classification/Point_set_neighborhood.h>

#include <CGAL/Epick_d.h>
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_d.h>

#include <CGAL/IO/read_ply_points.h>

#include <fstream>
#include <iostream>
#include <utility>

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


int main (int argc, char** argv)
{
  constexpr unsigned int nb_neighbors = 6;

  Point_range points;
  CGAL::Identity_property_map<Point_3> point_map;
  // Property maps are used to handle different types of input while
  // still getting CGAL Point_3 objects. In that case, I use a vector
  // of CGAL Point_3, so there's nothing to do, hence "identity"

  // TODO 1. Read your input points using either:
  if(argc != 2){
        std::cout << "execute program using: " << "./gret-sdp_with_OpenGR" << " <config/file/path>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string fname(argv[1]);

    std::ifstream input(fname);
    if (!input ||
        !CGAL::read_ply_points(input, std::back_inserter(points),
              CGAL::parameters::point_map (CGAL::Identity_property_map<Point_3>())))
    {
      std::cerr << "Error: cannot read file " << fname << std::endl;
      return EXIT_FAILURE;
    }
    input.close();

    for (const Point_3& point : points)
      std::cout << point.x() << ", " << point.y() << ", " << point.z() << std::endl;
    

  // This creates a 3D neighborhood + computes eigenvalues
  Neighborhood neighborhood (points, point_map);
  Eigen_analysis eigen = Eigen_analysis::create_from_point_set
    (points, point_map, neighborhood.k_neighbor_query(nb_neighbors));
  // If you want to use a fixed radius instead of KNN, use:
  // Eigen_analysis eigen = Eigen_analysis::create_from_point_set
  //   (points, point_map, neighborhood.sphere_neighbor_query(radius));


  // TODO 2. Populate features with the eigenvalues for each point
  Feature_range features;
  features.reserve (points.size());
  for (std::size_t i = 0; i < points.size(); ++ i)
  {
    // use Local_eigen_analysis::eigenvalue()
  }

  // This constructs a KD tree in N dimensions (here it's 3 but it
  // could be any dimension). Here, I'm using a trick: the KD tree
  // stores the indices of the points instead of the points themselves
  // (counting iterator just creates a range (0, 1, 2, ..., n)). This
  // allows, when querying the KD tree, to get the *index* of the
  // closest point (instead of the coordinates of the point itself,
  // which would be useless for you).
  Point_d_map point_d_map = CGAL::make_property_map(features);
  Kd_tree tree (boost::counting_iterator<std::size_t>(0),
                boost::counting_iterator<std::size_t>(features.size()),
                Splitter(),
                Search_traits(point_d_map));
  Distance dist (point_d_map);
  tree.build();

  std::vector<std::size_t> closest_point_in_feature_space (points.size());

  // TODO 3. Find closest points in feature spaces (I imagine you
  // should have one KD Tree per point cloud)
  for (std::size_t i = 0; i < points.size(); ++ i)
  {
    const Point_d& features_of_point_i = features[i];

    // Do the nearest neighbor query
    Knn knn (tree, // using the tree
             features_of_point_i, // for query point i
             1, // searching for 1 nearest neighbor only
             0, true, dist); // default parameters

    // You get the index of the nearest point. If you constructed the
    // KD tree directly on the feature point (without counting
    // iterator), you would have a Point_d type returned.
    std::size_t index_of_nearest_neighbor
      = knn.begin()->first;
  }

  // TODO 4. Call CGAL wrapper with feature


  return EXIT_SUCCESS;
}
