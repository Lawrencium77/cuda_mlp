# Cuda MLP
Training an MLP on MNIST in raw CUDA/C++.

## TODO

* Linting.
* Consider all of the synchronisation statements. Are these slowing me down loads? Will it require a PyTorch-style caching allocator?
* Proper tests.
* Plotting.
* Faster kernels, e.g. tiled matmuls.
