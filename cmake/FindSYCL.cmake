include(FindPackageHandleStandardArgs)

set(paths
    /usr
    /usr/local
)

find_path(SYCL_INCLUDE_DIR
    NAMES
        sycl.hpp
    PATHS
        ${paths}
    PATH_SUFFIXES
        include
)

find_package_handle_standard_args(SYCL
    SYCL_DEFAULT_MSG
    SYCL_INCLUDE_DIR
)
