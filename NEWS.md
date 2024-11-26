# mlr3torch dev

* perf: Use a faster image loader
* feat: Add parameter `num_interop_threads` to `LearnerTorch`
* feat: Add adaptive average pooling
* feat: Added `n_layers` parameter to MLP
* feat(perf): Added support for faster optimizers and trace-jitting,
  which can yield considerable performance improvements

# mlr3torch 0.1.2

* Don't use deprecated `data_formats` anymore
* Added `CallbackSetTB`, which allows logging that can be viewed by TensorBoard.

# mlr3torch 0.1.1

* fix(preprocessing): regarding the construction of some `PipeOps` such as `po("trafo_resize")`
  which failed in some cases.
* fix(ci): tests were not run in the CI
* fix(learner): `LearnerTabResnet` now works correctly
* Fix that tests were not run in the CI
* feat: added the `nn()` helper function to simplify the creation of neural network
  layers

# mlr3torch 0.1.0

* Initial CRAN release
