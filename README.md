# Cuda MLP
Training an MLP on MNIST in raw CUDA/C++.

## TODO

* Get the loss going down!
    * log-sum exp trick for softmax + CE
* I've a model that seems to run fwd passes fine. Next, update the run_fwd.cpp such that it computes CE loss.

### Later On

* Logging.
* Faster operations, e.g. tiled matmuls.
