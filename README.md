# Cuda MLP
Training an MLP on MNIST in raw CUDA/C++.

## TODO

* Linting.
* Switch to CMake.
* log-sum exp trick for softmax + CE/
* Why are larger batch sizes giving NaNs?
* Consider all of the synchronisation statements. Are these slowing me down loads? Will it require a PyTorch-style caching allocator?
* Proper tests.
* Plotting.
* Faster kernels, e.g. tiled matmuls.
