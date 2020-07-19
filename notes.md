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