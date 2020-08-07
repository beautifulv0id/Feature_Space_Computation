#include <vector>
#include <iostream>
#include <boost/range/iterator_range.hpp>
#include <CGAL/boost/graph/Named_function_parameters.h>
#include <CGAL/boost/graph/named_params_helper.h>

using namespace std;

int main() {
  vector<int> vals1 = {1, 2 , 3, 4, 5};
  vector<int> vals2 = {1, 2 , 3, 4, 5};

  vector<vector<int>> vals_range;
  vals_range.push_back(vals1);
  vals_range.push_back(vals2);


  auto unary_function = [](const auto& arg) { return arg + 1; };

  auto boost_it_range = 
  boost::make_iterator_range(
      boost::make_transform_iterator (vals1.begin(), unary_function),
      boost::make_transform_iterator (vals1.end(),   unary_function));

  for(auto val : boost_it_range)
    cout << val << endl;
}