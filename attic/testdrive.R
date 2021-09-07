# Testing tabnet ----
library(tabnet)
# data(ames, package = 'modeldata')
german_credit <- mlr3::tsk("german_credit")$data()

fit <- tabnet_fit(credit_risk ~ ., data = german_credit, epochs = 10)
predict(fit, german_credit[1:10,])
predict(fit, german_credit[1:10,], type = "prob")


# mlr3torch tabnet --------------------------------------------------------

library(mlr3)
library(mlr3torch)

task <- tsk("german_credit")

# Instantiate Learner
lrn = LearnerClassifTorchTabnet$new()

# Set Learner Hyperparams
lrn$param_set$values$epochs = 10


# Train and Predict
tictoc::tic()
lrn$train(task)
tictoc::toc()

preds <- lrn$predict(task)

preds$confusion
preds$score(msr("classif.acc"))

lrn$predict_type <- "prob"
preds_prob <- lrn$predict(task)

# preds$score(msr("time_predict"))
# preds$score(msr("time_train"))


# mlr3keras tabnet --------------------------------------------------------
# remotes::install_github('mlr-org/mlr3keras')
reticulate::use_condaenv("mlr3keras", required = TRUE)

# keras::install_keras("conda", tensorflow="2.4", envname="mlr3keras")
# reticulate::conda_install("mlr3keras", packages = "tabnet", pip = TRUE)

library(mlr3)
library(mlr3keras)
task <- tsk("german_credit")
keras_tabnet <- LearnerClassifTabNet$new()

keras_tabnet$param_set$values$epochs = 10

# Train and Predict
tictoc::tic()
keras_tabnet$train(task)
tictoc::toc()

preds <- keras_tabnet$predict(task)
preds$confusion
preds$score(msr("classif.acc"))


# Comparison --------------------------------------------------------------
library(mlr3)
library(mlr3torch)
library(mlr3keras)
reticulate::use_condaenv("mlr3keras", required = TRUE)

# task <- tsk("german_credit")

train_torchtabnet <- function(task, epochs = 10L) {
  torch_tabnet <- LearnerClassifTorchTabnet$new()

  ## torch params ----
  torch_tabnet$param_set$values$epochs = epochs
  torch_tabnet$param_set$values$batch_size = 256L
  torch_tabnet$param_set$values$num_steps = 2
  # N_d and N_a
  torch_tabnet$param_set$values$decision_width = 8
  torch_tabnet$param_set$values$attention_width = 8

  torch_tabnet$train(task = task)

  cat("torch task acc", torch_tabnet$predict(task)$score(msr('classif.acc'))[[1]])
}

train_kerastabnet <- function(task, epochs = 10L) {

  keras_tabnet <- LearnerClassifTabNet$new()

  ## keras params ----
  keras_tabnet$param_set$values$epochs = epochs
  keras_tabnet$param_set$values$batch_size = 256L
  keras_tabnet$param_set$values$num_decision_steps = 2
  # N_d and N_a
  keras_tabnet$param_set$values$output_dim = 8
  keras_tabnet$param_set$values$feature_dim = 8

  keras_tabnet$train(task = task)

  cat("keras task acc: ", keras_tabnet$predict(task)$score(msr('classif.acc'))[[1]])
}


## benchmark ----
benchres <- microbenchmark::microbenchmark(
  train_torchtabnet(tsk("german_credit"), epochs = 10L),
  train_kerastabnet(tsk("german_credit"), epochs = 10L),
  times = 10
)

saveRDS(benchres, file = "attic/benchres_german-credit.rds")

ggplot2::autoplot(benchres)

# pushoverr::pushover_normal('benchmark done')


