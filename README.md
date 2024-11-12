# Cuda MLP

Training an MLP on MNIST in raw CUDA/C++.

## Dependencies

Plotting loss curves requires [matplotlib](https://pypi.org/project/matplotlib/) and [seaborn](https://pypi.org/project/seaborn/).
<br>`wget` is required to download MNIST.

## TODO

* Linting.
* Check allocator for bugs/sane implementations.
* Allocate memory upfront, at the start of training.
* Always check error code from CUDA ops.
* Multiple CUDA streams.
* Proper tests.
* Faster kernels, e.g. tiled matmuls.
* Add more detail to the comparisons plot.

## Comparison

The following plot shows training loss for my implementation, and PyTorch. The performance is almost identical.


![](assets/comparison.png)

## The Need for a Custom allocator

Whenever we launch a CUDA kernel, we must allocate GPU memory for the output tensor with a call to `cudaMalloc`. This is a blocking call; it forces a synchronisation between host and device.

This is disastrous from a GPU utilisation perspective. To overcome this, the popular deep learning frameworks maintain their own "pool" of GPU memory which is managed internally, thus avoiding repeated calls to `cudaMalloc` and `cudaFree`. Examples include [PyTorch's CUDA Caching Allocator](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html) and [Tensorflow's BFC Allocator](https://github.com/sourcecode369/tensorflow-1/blob/master/tensorflow/core/common_runtime/bfc_allocator.cc).

Here's an illustration of the problem:

![](assets/nsight_image.png)

In this diagram, we run a matrix multiply following by a ReLU. In the CUDA API stream we see `cudaMalloc` calls in red, and kernel invocations in blue. The poor CUDA stream saturation is clear to see.
