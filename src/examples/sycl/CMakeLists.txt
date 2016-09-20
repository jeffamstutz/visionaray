find_package(GLUT REQUIRED)
find_package(OpenCL REQUIRED)
find_package(SYCL REQUIRED)

visionaray_use_package(GLUT)
visionaray_use_package(OpenCL)
visionaray_use_package(SYCL)

set(EX_SYCL_SOURCES
    main.cpp
)

visionaray_add_executable(sycl
    ${EX_SYCL_SOURCES}
)
