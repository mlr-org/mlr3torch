# Test without helper.debugging.R -----------------------------------------
library(mlr3)
library(mlr3torch)
# Load test scaffolding without helper_debugging.R
lapply(list.files(system.file("testthat", package = "mlr3"), pattern = "^helper.*\\.[rR]$", full.names = TRUE)[-2], source)
learner <- lrn("classif.torch.tabnet")

# Extracted from run_experiment()
tasks <- generate_tasks(learner, N = 30L)

# Pick the test which fails
task <- tasks[["feat_all_binary"]]

# Training works fine
learner$train(task)

# No errors
learner$errors


# With helper_debugging.R --------------------------------------------------
library(mlr3)
library(mlr3torch)

# Load mlr3 testthat scaffolding
lapply(list.files(system.file("testthat", package = "mlr3"), pattern = "^helper.*\\.[rR]$", full.names = TRUE), source)

learner <- lrn("classif.torch.tabnet")

# Extracted from run_experiment()
tasks <- generate_tasks(learner, N = 30L)

# Pick the test which fails
task <- tasks[["feat_all_binary"]]

# Error is triggered on training
learner$train(task)

