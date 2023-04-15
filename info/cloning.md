# Cloning

Some special care needs to be taken when verifying the correctness of deep clones.
Because torch wraps the R6 classes in functions, the `expect_deep_clone()` function from mlr3pipelines does not work.
These generators should not be modified anyway, so we do NOT clone them.
