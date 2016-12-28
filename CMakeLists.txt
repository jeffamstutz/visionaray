project(visionaray)


cmake_minimum_required(VERSION 2.8.8)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH}
    "${PROJECT_SOURCE_DIR}/cmake"
    "${PROJECT_SOURCE_DIR}/cmake/modules"
)

include(CTest)
enable_testing()
include(VisionarayCompileFailureTest)

include(VisionarayAddTarget)
include(VisionarayUseFramework)
include(VisionarayUsePackage)


#--------------------------------------------------------------------------------------------------
# Setup default build type and set valid build types
#

SET(CONFIGURATION_TYPES "Debug;Release;RelWithDebInfo")
IF (WIN32)
  IF (NOT OSPRAY_DEFAULT_CMAKE_CONFIGURATION_TYPES_SET)
    SET(CMAKE_CONFIGURATION_TYPES "${CONFIGURATION_TYPES}"
        CACHE STRING "List of generated configurations." FORCE)
    SET(OSPRAY_DEFAULT_CMAKE_CONFIGURATION_TYPES_SET ON
        CACHE INTERNAL "Default CMake configuration types set.")
  ENDIF()
ELSE()
  IF(NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the build type." FORCE)
  ENDIF()
  SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS ${CONFIGURATION_TYPES})
ENDIF()

#--------------------------------------------------------------------------------------------------
# CMake policies to control backward compatible behavior
#

SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)


#--------------------------------------------------------------------------------------------------
# CMake policies to control backward compatible behavior
#

# Old behavior: COMPILE_DEFINITIONS_<CONFIG>
# New behavior: COMPILE_DEFINITIONS + generator expression
if(POLICY CMP0043)
    cmake_policy(SET CMP0043 NEW)
endif()


#---------------------------------------------------------------------------------------------------
# Visionaray Options
#

option(BUILD_SHARED_LIBS "Build all libraries as shared libraries instead of static" OFF)
option(VSNRAY_ENABLE_WARNINGS "Enable all warnings" ON)
option(VSNRAY_ENABLE_PEDANTIC "Compile with pedantic enabled (Ignored if warnings are disabled)" ON)
option(VSNRAY_ENABLE_CUDA "Use CUDA, if available" ON)
option(VSNRAY_ENABLE_EXAMPLES "Build the programming examples" OFF)
option(VSNRAY_ENABLE_QT5 "Use Qt5, if available" OFF)
option(VSNRAY_ENABLE_VIEWER "Build the vsnray-viewer program" ON)
option(VSNRAY_ENABLE_UNITTESTS "Build unit tests" OFF)


#---------------------------------------------------------------------------------------------------
# All VSNRAY_* options have to be specified before this include
#

include(VisionarayHandleOptions)

add_subdirectory(src)
add_subdirectory(test)
