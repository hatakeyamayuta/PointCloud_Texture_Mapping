cmake_minimum_required(VERSION 3.5)
project(GL_sample_thread)

set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_FLAGS "-o2 -pthread" )

find_package(PkgConfig REQUIRED)
find_package(X11 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenCV REQUIRED)
find_package(PCL 1.11.1 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories(${GLUT_INCLUDE_DIRS})
add_executable(GL_sample main.cpp trackball.cpp)
target_link_libraries(GL_sample ${X11_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBS} ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES})
