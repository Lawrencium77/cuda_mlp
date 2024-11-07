# Cuda MLP

Training an MLP on MNIST in raw CUDA/C++.

## Dependencies

Plotting loss curves requires [matplotlib](https://pypi.org/project/matplotlib/).
`wget` is required to download MNIST.

## TODO

* Linting.
* Check that my implementation matches PyTorch.
* Consider all of the synchronisation statements:
     Are these slowing me down loads? Do some profiling.
     Will it require a PyTorch-style caching allocator?
* Proper tests.
* Faster kernels, e.g. tiled matmuls.
