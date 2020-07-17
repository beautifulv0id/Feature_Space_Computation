
set(OpenGR_ROOT_DIR "$ENV{OpenGR_ROOT_DIR}" CACHE PATH "OpenGR root directory.")
message("Looking for OpenGR in ${OpenGR_ROOT_DIR}")


set(OpenGR_INCLUDE_DIR_	"${OpenGR_ROOT_DIR}/include")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenGR OpenGR_INCLUDE_DIR)

if(OpenGR_FOUND)
    message("—- Found OpenGR under ${OpenGR_INCLUDE_DIR_}")
    set(OpenGR_INCLUDE_DIR ${OpenGR_INCLUDE_DIR_})
endif(OpenGR_FOUND)

mark_as_advanced(OpenGR_INCLUDE_DIR_)
