# Testing tabnet
library(tabnet)
data(ames, package = 'modeldata')

fit <- tabnet_fit(Sale_Price ~ ., data = ames, epochs = 1)



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
# preds$score(msr("time_predict"))
# preds$score(msr("time_train"))


# mlr3keras tabnet --------------------------------------------------------

# library(reticulate)
# Execute and restart R afterwards
# reticulate::conda_create(
#   envname = "mlr3keras",
#   packages = c("pandas", "python=3.8")
# )
# keras::install_keras("conda", tensorflow="2.5", envname="mlr3keras")
# reticulate::conda_install("mlr3keras", packages = "tabnet", pip = TRUE)

reticulate::use_condaenv("mlr3keras", required = TRUE)
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
