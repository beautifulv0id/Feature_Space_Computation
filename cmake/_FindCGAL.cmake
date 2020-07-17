
set(CGAL_ROOT_DIR "$ENV{CGAL_ROOT_DIR}" CACHE PATH "CGAL root directory.")
message("Looking for CGAL in ${CGAL_ROOT_DIR}")

set(CGAL_INCLUDE_DIR_	"${CGAL_ROOT_DIR}/include")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CGAL CGAL_INCLUDE_DIR)

if(CGAL_FOUND)
    message("—- Found CGAL under ${CGAL_INCLUDE_DIR_}")
    set(CGAL_INCLUDE_DIR ${CGAL_INCLUDE_DIR_})
endif(CGAL_FOUND)

mark_as_advanced(CGAL_INCLUDE_DIR_)
