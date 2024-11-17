# Cuda MLP

This project trains an MLP on MNIST in pure CUDA/C++. It aims to be simple and quick.

## Accuracy

The following plot shows training loss for my implementation, and PyTorch. The performance is almost identical.


![](assets/comparison.png)

## Speed

A single epoch (including validation) takes around 750ms on an H100.

### The Need for a Custom Memory Allocator

Whenever launching a CUDA kernel, we must allocate GPU memory for the output tensor with a call to `cudaMalloc`. This forces a host-device synchronisation.

This is bad for GPU utilisation. As a workaround, popular deep learning frameworks manage their own "pool" of GPU memory, thus avoiding repeated calls to `cudaMalloc`/`cudaFree`. Examples include [PyTorch's CUDA Caching Allocator](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html) and [Tensorflow's BFC Allocator](https://github.com/sourcecode369/tensorflow-1/blob/master/tensorflow/core/common_runtime/bfc_allocator.cc).

Here's an illustration of the problem:

![](assets/nsight_image.png)

In this diagram, we run a matmul followed by a ReLU. In the CUDA API stream we see `cudaMalloc` calls in red, and kernel invocations in blue.

After implementing our custom allocator, CUDA kernels are launched asynchronously:

![](assets/nsight_image2.png)

Logging each call to `cudaMalloc` shows that it is only called during the first epoch:

```
Calling allocate, cudaMalloc
...
Calling allocate, cudaMalloc
Epoch 1/10
Calling allocate, cudaMalloc
...
Calling allocate, cudaMalloc
Validation Loss after epoch 1: 0.167855
Validation Acc after epoch 1: 94.84%
Epoch 2/10
Validation Loss after epoch 2: 0.109865
Validation Acc after epoch 2: 96.42%
Epoch 3/10
...
```

## Dependencies

I've tried to keep the number of dependencies as small as possible. Currently, the required dependencies are:

* `wget` (to download MNIST).
* [matplotlib](https://pypi.org/project/matplotlib/) and [seaborn](https://pypi.org/project/seaborn/) (for plotting loss curves).

## TODO

* Compare our throughput with PyTorch on various hardware.
* Faster kernels, (tiled matmul, fused matmul-ReLU, etc.)
* Support multiple dtypes (we currently only support FP32)
* Add more detail to loss plot.
* More thorough unit tests.
* Linting
* Build with CMake
