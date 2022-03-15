library(mlr3)
library(mlr3torch)

# Load test scaffolding without helper_debugging.R
lapply(list.files(system.file("testthat", package = "mlr3"), pattern = "^helper.*\\.[rR]$", full.names = TRUE)[-2], source)
learner <- lrn("classif.torch.tabnet")

# Extracted from run_experiment()
tasks <- generate_tasks(learner, N = 30L)

# Pick the test which fails
task <- tasks[["feat_all_binary"]]

lrn = lrn("classif.torch.tabnet", epochs = 1)

lrn$train(task)

tabnet::tabnet_fit(
  x = task$data(cols = task$feature_names),
  y = task$data(cols = task$target_names)
)

# tabnet gives two importance scores per logical variable value
lrn$importance()
task$feature_names

# Different learner result for reference ----------------------------------
library(mlr3learners)
lrnranger <- lrn("classif.ranger", importance = "impurity_corrected")
lrnranger$train(task)

lrnranger$importance()
task$feature_names


x <- tabnet::tabnet_fit(target ~ ., data = task$data()[, logical := as.numeric(logical)])
x$fit$importances


# reprex for tabnet -------------------------------------------------------

library(tabnet)

set.seed(2)
# Training data with logical feature --------------------------------------
xdat <- tibble::tibble(
  feat_factor = factor(sample(letters, 100, replace = TRUE)),
  feat_numeric = rnorm(100),
  feat_integer = sample(100, replace = TRUE),
  feat_logical = sample(c(TRUE, FALSE), 100, replace = TRUE),
  target = factor(sample(c("yes", "no"), 100, replace = TRUE))
)

model_fit <- tabnet_fit(target ~ ., data = xdat, epochs = 3)

# Distinct importance scores for TRUE and FALSE seem... odd
model_fit$fit$importances

# Recoded to integer ------------------------------------------------------
xdat$feat_logical <- as.integer(xdat$feat_logical)

model_fit2 <- tabnet_fit(target ~ ., data = xdat)

# Importance scores as expected, one per input feature
model_fit2$fit$importances

