# Created by the script cgal_create_CMakeLists
# This is the CMake script for compiling a set of CGAL applications.

project( test_cmake )

cmake_minimum_required(VERSION 2.8.11)

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/")


###############################################################################
# TARGETS
###############################################################################
set(targets
gret_sdp_wrapper_test
feature_matching_evaluation
)




###############################################################################
# GENERATED PART
###############################################################################

# Libraries and flags
set(project_linked_libraries)
set(project_compilation_definitions)


# MOSEK
find_package(MOSEK)
if ( MOSEK_FOUND )
include_directories(${MOSEK_INCLUDE_DIR})
list(APPEND project_linked_libraries ${MOSEK_LIBRARIES})
else()
message(STATUS "This project requires the MOSEK library, and will not be compiled.")
return()  
endif()

# OpenGR
find_package(OpenGR)
if ( OpenGR_FOUND AND OpenGR_USE_MOSEK)
  include_directories(${OpenGR_INCLUDE_DIR})
  list(APPEND project_compilation_definitions OpenGR_USE_MOSEK)
  message(STATUS "FOUND Open_GR ${OpenGR_INCLUDE_DIR}")
else()
  message(STATUS "This project requires the OpenGR library, and will not be compiled.")
  return()  
endif()

# CGAL and its components
find_package( CGAL QUIET COMPONENTS  )
if ( CGAL_FOUND )
  add_dependencies(CGAL OpenGR) 
  include_directories(${CGAL_INCLUDE_DIR})
  message(STATUS "FOUND CGAL")
else() 
  message(STATUS "This project requires the CGAL library, and will not be compiled.")
  return()  
endif()

# include helper file
include( ${CGAL_USE_FILE} )

# Boost and its components
find_package(Boost 1.54 COMPONENTS system filesystem  REQUIRED)

if ( Boost_FOUND )
  include_directories(${Boost_INCLUDE_DIRS})
  list(APPEND project_linked_libraries ${PCL_LIBRARIES})
  message(STATUS "Found BOOST")
else()
  message(STATUS "This project requires the Boost library, and will not be compiled.")
  return()  
endif()

# point cloud library
find_package( PCL 1.2 REQUIRED)
if ( PCL_FOUND )
  include_directories( ${PCL_INCLUDE_DIRS} )
  add_definitions( ${PCL_DEFINITIONS} )
  link_directories( ${PCL_LIBRARY_DIRS} )
  message(STATUS ${PCL_LIBRARIES})
  list(APPEND project_linked_libraries ${PCL_LIBRARIES})
  message(STATUS "FOUND PCL")
endif()


include( CGAL_CreateSingleSourceCGALProgram )

# Eigen
find_package(Eigen3 3.1.0) #(requires 3.1.0 or greater)
if (EIGEN3_FOUND)
  include( ${EIGEN3_USE_FILE} )
  message(STATUS "FOUND EIGEN")
else()
  message(STATUS "NOTICE: This project requires the Eigen library, and will not be compiled.")
  return()
endif()

# Creating targets with correct libraries and flags
foreach(target ${targets})
  create_single_source_cgal_program( "${target}.cpp" )
  if(TARGET ${target})
    target_link_libraries(${target} PUBLIC ${project_linked_libraries})
    target_compile_definitions(${target} PUBLIC ${project_compilation_definitions})
  endif()
endforeach()