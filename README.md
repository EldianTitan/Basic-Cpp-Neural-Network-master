# Basic-Cpp-Neural-Network

## Description

This project is a basic deep neural network implemented in C++. The network is trained to recognize handwritten digits. Training and testing data sets are from the [MNIST Database](http://yann.lecun.com/exdb/mnist/).

## Building

### Requirements

In order to build this project, `cmake` is needed along with any C++ compiler.

### Build Steps

Navigate to the root directory of the project, open a terminal, and execute the following:

```
mkdir build
cd ./build
cmake ..
```

### Running

Before running the project, you must replace `<Path_to_project>` in `main.cpp` with the path of the project's root directory.

You can run the project by navigating to the `build` directory, and running `cmake --build .`. If any cmake-compatible IDEs are installed on your system, cmake will automatically detect them and build the project for these environments. 

### Note

This project is meant for demonstration purposes ONLY, and is not very performant. Thus, it is recommended that you build your project in Release mode when training the network.