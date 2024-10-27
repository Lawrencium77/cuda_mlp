# Cuda MLP
Training an MLP on MNIST in raw CUDA/C++.

## TODO

* Sort out dimensions for softmax and CE loss. I think things are transposed.
* I've a model that seems to run fwd passes fine. Next, update the run_fwd.cpp such that it computes CE loss.

### Later On

* Backprop.
* Logging.
* Faster operations, e.g. tiled matmuls.
