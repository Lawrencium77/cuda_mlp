# Cuda MLP
Training an MLP on MNIST in raw CUDA/C++.

## TODO

* Logging.
* Consider all of the synchronisation statements. Are these slowing me down loads?
* Could make lots of const args everwhere.
* Surely there is a refactor required for lots of parts of the codebase.
* Proper tests.
* log-sum exp trick for softmax + CE/
* Faster kernels, e.g. tiled matmuls.
