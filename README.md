# Cuda MLP
Training an MLP on MNIST in raw CUDA/C++.

## TODO

* Linting.
* Consider all of the synchronisation statements. Are these slowing me down loads?
* Could make lots of const args everwhere.
* Why are larger batch sizes giving NaNs?
* Switch to CMake.
* Surely there is a refactor required for lots of parts of the codebase.
* Proper tests.
* Plotting.
* log-sum exp trick for softmax + CE/
* Faster kernels, e.g. tiled matmuls.
