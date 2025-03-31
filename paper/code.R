set.seed(42)

library(mlr3)
task <- tsk("mtcars")
learner <- lrn("regr.rpart")
split <- partition(task, ratio = 2/3)
learner$train(task, split$train)
pred <- learner$predict(task, split$test)
measure <- msr("regr.rmse")
pred$score(measure)

library(mlr3pipelines)
graph_learner = as_learner(po("pca") %>>% lrn("regr.rpart))