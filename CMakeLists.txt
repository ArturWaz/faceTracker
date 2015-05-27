cmake_minimum_required(VERSION 3.0.2)


set(PROJECT_NAME faceTracker)
project(${PROJECT_NAME})

#set(CMAKE_PREFIX_PATH "C:\\Qt\\Qt5.4.1\\5.4\\mingw491_32")
#set(CMAKE_PREFIX_PATH CMAKE_PREFIX_PATH "C:\\boost\\")
set(CMAKE_PREFIX_PATH CMAKE_PREFIX_PATH "F:\\_compilers\\_lib\\opencv\\build")


set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_SHARED_LINKER_FLAGS "-static")
if (NOT MSVC)
    set(CMAKE_CXX_FLAGS "-std=c++11")
endif (NOT MSVC)
if (UNIX)
    set( CMAKE_CXX_FLAGS "-pthread ${CMAKE_CXX_FLAGS}" )
endif (UNIX)


find_package(OpenCV REQUIRED)


add_executable(${PROJECT_NAME} main.cpp rs232.c)
target_link_libraries (${PROJECT_NAME} ${OpenCV_LIBS})
