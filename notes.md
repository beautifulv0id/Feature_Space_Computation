# USE FEATURE MATCHING TO CONSTRUCT MEMBERSHIP GRAPH

## INPUT
- `m`: number of point clouds
- `point_clouds`: range of point clouds

## OUTPUT
- `indexed_point_clouds`: range of indexed point clouds

## COMPUTATIONS
- compute `feature_range` for each point cloud
- compute nearest neighbor for each point in point cloud

### restrictions
- points can not share the same correspondency within one point cloud

### refinement
- check if correspondency can be matched back
- check if correspondency distance is under certain threshold?
- check if distance between best and second best correspondency is under certain threshold
- for correspondences that cause "collision" take those with the least standard derivation

### naive apprach
- compute for first point cloud:
    - nearest neighbor for every point in other point clouds
    - for every neighbor:
        - only add if it can be matched back
        - probably rejects loads of correspondences

- needed:
    - kd tree for every pc in `kd_tree_range`
    - features for every pc in `feature_range`
    - 

### NOW 
- ADD KD TREE RANGE


```cpp
// hash maps
  std::vector<Map> map_range;
  map_range.reserve(num_point_clouds);
  for (int i = 0; i < num_point_clouds; i++)
    map_range.push_back(Map());
  
  // for every point cloud
  for (size_t current_pc_index = 0; current_pc_index < num_point_clouds; current_pc_index++)
  {
    Kd_tree& current_tree = *tree_range[current_pc_index]; 
    Feature_range& current_features = feature_ranges[current_pc_index];
    Map& current_map = map_range[current_pc_index];
    Map_iterator current_map_it;
    uint current_pc_start_index = pc_start_index[current_pc_index];


    // compare to every other point cloud k
    for (size_t other_pc_index = 0; other_pc_index < num_point_clouds; other_pc_index++)
    {
      if(other_pc_index != current_pc_index){
        Kd_tree& other_tree = *tree_range[other_pc_index]; 
        Feature_range& other_features = feature_ranges[other_pc_index];
        Map& other_map = map_range[other_pc_index];
        Map_iterator other_map_it;
        uint other_pc_start_index = pc_start_index[other_pc_index];

        
        // every point in current point cloud
        for (size_t current_point_index = 0; current_point_index < point_ranges[current_pc_index].size(); current_point_index++)
        {
            const Point_d& current_point_features = current_features[current_point_index];
            // Do the nearest neighbor query
            Knn other_knn (other_tree, // using the tree
                    current_point_features, // for query point i
                    1, // searching for 1 nearest neighbor only
                    0, true, distances[other_pc_index]); // default parameters

            // index of nearest neighbor
            std::size_t index_of_nearest_neighbor
              = other_knn.begin()->first;
            // distance between points
            double current_dist = other_knn.begin()->second;


            // to look if point has been added already
            current_map_it = current_map.find(std::make_pair(other_pc_index, current_point_index));

            // if point wasnt matched before
            if(current_map_it == current_map.end()){
              // search other point in map
              other_map_it = other_map.find(std::make_pair(current_pc_index, index_of_nearest_neighbor));

              // if nn wasn't matches by point within same point cloud before
              if(other_map_it == other_map.end()){
                const Point_d& nearest_neighbor_features = other_features[index_of_nearest_neighbor];
                Knn current_knn (current_tree, // using the tree
                    nearest_neighbor_features, // for query point i
                    1, // searching for 1 nearest neighbor only
                    0, true, distances[current_pc_index]); // default parameters
                // index of nearest neighbor
                std::size_t current_index_of_nearest_neighbor
                  = current_knn.begin()->first;

                // if correspondence matches back add correspondence
                if(current_index_of_nearest_neighbor == current_pc_index + current_point_index){
                  // update graph
                  addCorrespondence( 
                    feature_graph,
                    other_pc_start_index + index_of_nearest_neighbor, 
                    current_pc_start_index + current_point_index
                  );

                  // update maps
                  other_map[std::make_pair(current_pc_index, index_of_nearest_neighbor)]
                  = std::make_pair(current_pc_start_index + current_point_index, current_dist);
                  
                  current_map[std::make_pair(other_pc_index, current_point_index)] 
                    = std::make_pair(other_pc_start_index + index_of_nearest_neighbor, current_dist);
                }
              } 
              // if nn was matched by point within same point cloud before
              else {
                  Map_value_type prev_idx_with_dist = other_map_it->second;
                  uint prev_idx = get<uint>(prev_idx_with_dist);
                  double prev_dist = get<double>(prev_idx_with_dist);
                  // if smaller: add current correspondence/ remove prev and update map
                  if(current_dist < prev_dist){                    
                    addCorrespondence( 
                      feature_graph,
                      other_pc_start_index + index_of_nearest_neighbor, 
                      current_pc_start_index + current_point_index
                    );
                    other_map[std::make_pair(current_pc_index, index_of_nearest_neighbor)]
                      = std::make_pair(current_pc_start_index + current_point_index, current_dist);
                    current_map[std::make_pair(other_pc_index, current_point_index)] 
                      = std::make_pair(other_pc_start_index + index_of_nearest_neighbor, current_dist);

                    removeCorrespondence(
                      feature_graph,
                      other_pc_start_index + index_of_nearest_neighbor,
                      prev_idx
                    );
                    current_map.erase(std::make_pair(other_pc_index, prev_idx));


                  }
              }
            }
        }
      }
    }
  }

```