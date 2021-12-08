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

