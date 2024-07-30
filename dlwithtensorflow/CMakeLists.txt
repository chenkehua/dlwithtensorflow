project(ocr CXX C)
cmake_minimum_required(VERSION 3.14)

SET(OPENCV_DIR "" CACHE PATH "Location of libraries")
#SET(ONNXLIB "/paddle-infer/20240510/paddle_openblas/paddle_inference/third_party/install/onnxruntime/lib/libonnxruntime.so" CACHE PATH "Location of libraries")
SET(ONNXLIB "~/Research/ckh/paddle-infer/test/20240510/paddle_openblas/paddle_inference/third_party/install/onnxruntime/lib/libonnxruntime.so" CACHE PATH "Location of libraries")

set(DEMO_NAME "ocr")
macro(safe_set_static_flag)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
endmacro()

if(NOT DEFINED OPENCV_DIR)
    message(FATAL_ERROR "please set OPENCV_DIR with -DOPENCV_DIR=/path/opencv")
endif()

if (WIN32)
  find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/build/ NO_DEFAULT_PATH)
else ()
  find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR}/share/OpenCV NO_DEFAULT_PATH)
endif ()
include_directories(${OpenCV_INCLUDE_DIRS})

if (WIN32)
    add_definitions("/DGOOGLE_GLOG_DLL_DECL=")
    set(CMAKE_C_FLAGS_DEBUG   "${CMAKE_C_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_C_FLAGS_RELEASE  "${CMAKE_C_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} /bigobj /MTd ${FLAG_OPENMP}")
    set(CMAKE_CXX_FLAGS_RELEASE   "${CMAKE_CXX_FLAGS_RELEASE} /bigobj /MT ${FLAG_OPENMP}")
    message("cmake c debug flags " ${CMAKE_C_FLAGS_DEBUG})
    message("cmake c release flags " ${CMAKE_C_FLAGS_RELEASE})
    message("cmake cxx debug flags " ${CMAKE_CXX_FLAGS_DEBUG})
    message("cmake cxx release flags " ${CMAKE_CXX_FLAGS_RELEASE})
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3 ${FLAG_OPENMP} -std=c++11")
    message("cmake cxx flags" ${CMAKE_CXX_FLAGS})
endif()

include_directories("${CMAKE_SOURCE_DIR}/")

include_directories("~/Research/ckh/paddle-infer/test/20240510/paddle_openblas/paddle_inference/third_party/install/onnxruntime/include")
link_directories("~/Research/ckh/paddle-infer/test/20240510/paddle_openblas/paddle_inference/third_party/install/onnxruntime/lib")

set(DEPS ${DEPS} ${OpenCV_LIBS})

set(SOURCE_FILES
    main_back.cpp
    text_angle_cls.cpp
    text_det.cpp
    text_rec.cpp
    layout_det.cpp
    table_det.cpp
    utility.cpp
)
include(FetchContent)

add_executable(${DEMO_NAME} ${SOURCE_FILES})
target_link_libraries(${DEMO_NAME} ${DEPS} ${ONNXLIB})
