#include <string>
#include <vector>
#include <CGAL/IO/read_ply_points.h>
#include <boost/filesystem.hpp>
#include <Eigen/Dense>
#include <unordered_map>
#include <functional>
#include <queue>

#include <thread>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>

#include <Eigen/Dense>

// struct that stores the index of a points:
// point cloud in pc_
// position within that point cloud in idx_
struct PointIndexPair {
  unsigned int pc_;
  unsigned int idx_;

  PointIndexPair(int pc, int idx) : pc_(pc), idx_(idx) {}
};

// stores the indexes of a correspondence and its total feature distance
struct Correspondences {
  private:
  double total_feature_distance;
  std::vector<PointIndexPair> correspondences;

  public:
  Correspondences() : total_feature_distance(0) {}

  void add(const PointIndexPair& correspondence){
    correspondences.emplace_back(correspondence.pc_, correspondence.idx_);
  }

  void add(int pc, int idx){
    correspondences.emplace_back(pc, idx);
  }

  void updateTotalDistance(const double dist){
    total_feature_distance += dist;
  }

  double getWeight() const {
    return total_feature_distance / (double) (size()*(size()-1)/2);
  }

  const PointIndexPair& operator[](int i) const {
    return correspondences[i];
  } 

  const int size() const {return correspondences.size(); }
  const std::vector<PointIndexPair>& vec() const  { return correspondences; }

  bool operator()(const Correspondences& lhs, const Correspondences& rhs)
                    { return lhs.getWeight()  > rhs.getWeight(); }
};



// hash map for pairs
struct hash_pair { 
    template <class T1, class T2> 
    size_t operator()(const std::pair<T1, T2>& p) const
    { 
        auto hash1 = std::hash<T1>{}(p.first); 
        auto hash2 = std::hash<T2>{}(p.second); 
        return hash1 ^ hash2; 
    } 
};

// map that helps to check if a src_point already has a corresponcence within target_point_cloud
// returns the target_point if correspondence exists
// key: (target_point_cloud, src_point)
// val: target_point
using Map = std::unordered_map<std::pair<uint, uint>, uint, hash_pair>;
using Map_iterator = typename Map::iterator;
using Map_const_iterator = typename Map::const_iterator;
using Map_value_type = typename Map::value_type::second_type;

// helper function to check if a correspondence already exists in target_point_cloud for src_point
bool correspondenceExists(const Map& map, const int pc, const int idx){
  return map.find(std::make_pair(pc, idx)) == map.end() ? false : true;
}

// updates hash maps for with the given correspondences
template <typename MapRange>
void updateMaps(const Correspondences& correspondences, MapRange& map_range){
  for (int i = 0; i < correspondences.size(); i++)
    for (size_t j = 0; j < correspondences.size(); j++)
      if(i != j)
        map_range[correspondences[i].pc_][std::make_pair(correspondences[j].pc_, correspondences[i].idx_)]
        = correspondences[j].idx_;
}


// extracts point clouds and transformations from a stanford config file
template <typename PointMap, typename Point_range, typename Transformation>
void extractPCAndTrFromStandfordConfFile(
        const std::string &confFilePath,
        std::vector<Transformation>& transforms,
        std::vector<Point_range>& pc_range,
        const PointMap& point_map
        ){
    using namespace boost;
    using namespace std;

    typedef Eigen::Transform<double, 3, Eigen::Affine> Transform;

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
                string inputfile = filesystem::path(confFilePath).parent_path().string()+string("/")+tokens[1];
                //VERIFY(filesystem::exists(inputfile) && filesystem::is_regular_file(inputfile));

                // build the Eigen rotation matrix from the rotation and translation stored in the files
                Eigen::Matrix<double, 3, 1> tr (
                            std::atof(tokens[2].c_str()),
                            std::atof(tokens[3].c_str()),
                            std::atof(tokens[4].c_str()));

                Eigen::Quaternion<double> quat(
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
        } 
        pc_file.close();
    }
}

// stores min, max and average distances of the computed correspondences to improve visualization
struct Distances {
  double min_, max_, average_;
  Distances(double min, double max, double average) :
    min_(min), max_(max), average_(average){};
};

typedef pcl::PointXYZ PointNT;
typedef pcl::PointCloud<PointNT> PointCloudT;


template <typename PointRange, typename Transformation>
pcl::PointCloud<pcl::PointXYZ>::Ptr CGAL2PCL_Point_Cloud(const PointRange& cgal_pcloud, const Transformation& transformation){
    using Point_3 = typename PointRange::value_type;
    PointCloudT::Ptr pcl_pcloud (new PointCloudT);
    for (auto& point : cgal_pcloud)
    {
        const Point_3& transformed_point = point.transform(transformation.inverse());
        PointNT pcl_point(transformed_point.x(), transformed_point.y(), transformed_point.z());
        pcl_pcloud->push_back(pcl_point);
    }
    return pcl_pcloud;
}

template <typename PointRange>
pcl::PointCloud<pcl::PointXYZ>::Ptr CGAL2PCL_Point_Cloud(const PointRange& cgal_pcloud){
    using Point_3 = typename PointRange::value_type;
    PointCloudT::Ptr pcl_pcloud (new PointCloudT);
    for (auto& point : cgal_pcloud)
    {
        PointNT pcl_point(point.x(), point.y(), point.z());
        pcl_pcloud->push_back(pcl_point);
    }
    return pcl_pcloud;
}

template <typename PointCloudRange, typename TransformationRange, typename CorrespondencesRange>
void visualizeCorrespondences(const PointCloudRange& point_clouds, const TransformationRange& transformations, const Distances& distances, 
                              const CorrespondencesRange& correspondences_range, bool translate = true, double dx = 0.2){
  using namespace std::chrono_literals;
  using Point_3 = typename PointCloudRange::value_type::value_type;

  if(point_clouds.size() != 2)
    throw std::runtime_error("correspondence visualization only availible for two point clouds");
  
  if(transformations.size() != 2)
    throw std::runtime_error("number of provided transformations has to be two");

  PointCloudT::Ptr pcloud1 = CGAL2PCL_Point_Cloud(point_clouds[0], transformations[0]);
  PointCloudT::Ptr pcloud2 = CGAL2PCL_Point_Cloud(point_clouds[1], transformations[1]);


  if(translate){
    Eigen::Matrix4d translation = Eigen::Matrix4d::Identity();
    translation.block<3,1>(0,3) = Eigen::Vector3d(dx, 0, 0);

    PointCloudT::Ptr pcloud2_translated(new PointCloudT());
    pcl::transformPointCloud(*pcloud2, *pcloud2_translated, translation);
    pcloud2 = pcloud2_translated;
  }

  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<PointNT> (pcloud1, "point cloud 1");
  viewer->addPointCloud<PointNT> (pcloud2, "point cloud 2");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "point cloud 1");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "point cloud 2");
  viewer->initCameraParameters();

  for (const Correspondences& correspondences : correspondences_range)
  {
    const PointIndexPair& A = correspondences[0];
    const PointIndexPair& B = correspondences[1];

    const PointNT src_idx = (*pcloud1)[A.idx_];
    const PointNT tgt_idx = (*pcloud2)[B.idx_];

    std::string lineID =  std::to_string(A.pc_) + " " + std::to_string(A.idx_)
                        + std::to_string(B.pc_) + " " + std::to_string(B.idx_);

    double current_dist = CGAL::squared_distance( point_clouds[A.pc_][A.idx_].transform(transformations[A.pc_].inverse()),
                                                    point_clouds[B.pc_][B.idx_].transform(transformations[B.pc_].inverse()));

    double scale = 0;
    if(current_dist < distances.average_)
      scale = (current_dist - distances.average_) / (- distances.average_);
     

    // Generate a random (bright) color
    double r = 255 * (1-scale);
    double g = 255 * scale;
    double b = 0;
    double max_channel = std::max(r, std::max(g, b));
    r /= max_channel;
    g /= max_channel;
    b /= max_channel;

    viewer->addLine<PointNT, PointNT>(src_idx, tgt_idx, r, g, b, lineID);
  }

  while (!viewer->wasStopped ())
  {
      viewer->spinOnce (100);
      std::this_thread::sleep_for(100ms);
  }
}
