cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
project(basicNN LANGUAGES CXX)

file(GLOB project_SRC
    "src/*.h"
	"src/*.cu"
    "src/*.cpp"
)

add_executable(basicNN ${project_SRC})

target_compile_features(basicNN PUBLIC cxx_std_11)