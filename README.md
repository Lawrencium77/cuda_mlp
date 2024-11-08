# Cuda MLP

Training an MLP on MNIST in raw CUDA/C++.

## Dependencies

Plotting loss curves requires [matplotlib](https://pypi.org/project/matplotlib/) and [seaborn](https://pypi.org/project/seaborn/).
<br>`wget` is required to download MNIST.

## TODO

* Linting.
* Consider all of the synchronisation statements:
     Are these slowing me down loads? Do some profiling.
     Will it require a PyTorch-style caching allocator?
* Proper tests.
* Faster kernels, e.g. tiled matmuls.
* Add more detail to the comparisons plot.

## Comparison

The following plot shows training loss for my implementation, and PyTorch. The performance is almost identical.


![](assets/comparison.png)
