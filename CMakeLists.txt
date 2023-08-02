cmake_minimum_required(VERSION 3.12)

project(cpp_tests)

# Add your C++ source file(s) here
add_executable(main main.cpp)

# Set the path to OpenCV installation
set(OpenCV_DIR "C:/Users/meto/Documents/opencv/build")

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Link OpenCV libraries to your executable
target_link_libraries(main PRIVATE ${OpenCV_LIBS})