# Cuda MLP with Custom Allocator

This is a simple project to train an MLP on MNIST using pure CUDA/C++. It implements two memory allocation strategies: a stream-ordered allocator leveraging CUDA's asynchronous API, and a custom pooling allocator.

## Dependencies

I've tried to keep the number of dependencies small. Currently, this project requires:

* `wget` (to download MNIST).
* [matplotlib](https://pypi.org/project/matplotlib/) and [seaborn](https://pypi.org/project/seaborn/) (for plotting loss curves).


## Usage

Build project with:

```bash
make clean && make all
```

Launch training run with:

```bash
build/train <path_to_config> # custom allocator
ALLOCATOR_TYPE=cuda build/train <path_to_config> # CUDA Stream Ordered Allocator
```

For accurate timing information, I suggest doing:

```bash
tools/throughput_test_wrapper.sh build/train <path_to_config>
```

This fixes GPU clock speeds, ensuring more repeatable timing information.


## Accuracy

The following plot shows training loss for my implementation, and PyTorch. The trajectories are almost identical.

![](assets/comparison.png)

## Speed

This project is not designed to be ultra-efficient. For example, its CUDA kernels are entirely unoptimised.

That being said, it's much quicker than PyTorch for small models. A single epoch (including validation) takes ~750ms on an H100 SXM for a model with 4 layers and a hidden dimension of 784. By comparison, PyTorch achieves ~3.6s per epoch (with `torch.compile` enabled and allowing for a warmup phase). **This custom implementation is >4x quicker**, primarily because it avoids any Python interpreter overhead.

## The Need for a Custom Memory Allocator

When launching a CUDA kernel, we must allocate GPU memory for the output tensor. The most naïve way to achieve this is a call to `cudaMalloc`, whis forces a host-device synchronisation.

This synchronisation blocks parallel CPU-GPU execution, severely impacting performance. As a workaround, popular deep learning frameworks manage their own "pool" of GPU memory, thus avoiding repeated calls to `cudaMalloc`/`cudaFree`. Examples include [PyTorch's CUDA Caching Allocator](https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html) and [Tensorflow's BFC Allocator](https://github.com/sourcecode369/tensorflow-1/blob/master/tensorflow/core/common_runtime/bfc_allocator.cc).

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

## Comparison to the Stream Ordered Memory Allocator
CUDA provides its own solution in the form of `CudaMallocAsync`/`cudaFreeAsync`. They call this the [Stream Ordered Memory Allocator](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/). The idea is identical to any custom memory allocator, except that the memory pool is handled by the CUDA driver rather than at the app level.

This project supports `CudaMallocAsync`/`cudaFreeAsync` in addition to a custom memory allocator. There are two points of note regarding the implementation.

First, unused memory accumulated in the pool is by default returned to the OS at each host-device synchronisation. This is clearly suboptimal since it'll prevent our pool from reaching a steady state. As a solution, we set `cudaMemPoolAttrReleaseThreshold` to its max value (`UINT64_MAX`). This guarantees memory won't be released prematurely.

Second, as Nvidia [describe](https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/#retaining_memory_in_the_pool):

> The pool threshold is just a hint. Memory in the pool can also be released implicitly by the CUDA driver to enable an unrelated memory allocation request in the same process to succeed.

For this reason, we must ensure that this project makes no calls to `cudaMalloc`/`cudaFree`. If we make allocation/free calls besides those handled by our memory allocator object, we see sporadic `cudaFree` calls which prevent our memory allocator from ever reaching a steady state.

Let's compare the throughputs of our two allocators for a model with `feat_dim=784`, `num_layers=4`:

```bash
# Custom Allocator
Epoch 1 took 844 ms
Epoch 2 took 843 ms
Epoch 3 took 843 ms

# CUDA Allocator
Epoch 1 took 882 ms
Epoch 2 took 885 ms
Epoch 3 took 884 ms
```

The two approaches are close but our custom allocator is ~40ms faster per epoch. Digging into this with `nsys`, we see the custom allocator is slightly faster for each fwd-bwd step. In this example, our custom allocator takes 510 μs:

![](assets/nsight_image3.png)

versus 521 μs for the CUDA allocator:

![](assets/nsight_image4.png)

The 21 μs discrepancy comes from a slightly longer idle time between consecutive kernels when using `cudaMallocAsync` compared to our custom allocator.

Though the exact cause would require further investigation, it seems that additional driver-level memory pool operations impact kernel scheduling efficiency. By contrast, our custom allocator operates at the application level, thus removing driver-managed pool operations from the critical path.

Of course, this performance difference becomes negligible with larger kernel sizes. Increasing `feat_dim` to `2048`, we see a ~30ms difference between allocator types - similar to above, despite the longer epoch times:

```bash
# Custom Allocator
Epoch 1 took 2422 ms
Epoch 2 took 2415 ms
Epoch 3 took 2413 ms

# CUDA Allocator
Epoch 1 took 2449 ms
Epoch 2 took 2442 ms
Epoch 3 took 2440 ms
```
