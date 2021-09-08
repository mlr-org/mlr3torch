## Regression ----
library(mlr3)
library(paradox)
library(mlr3torch)
library(mlr3pipelines)
library(mlr3tuning)

data("kc_housing", package = "mlr3data")
# Drop columns with missings and class POSIXct for simplicity
kc_housing <- kc_housing[, setdiff(names(kc_housing), c("date", "sqft_basement", "yr_renovated"))]
task_regr <- TaskRegr$new("kc_housing", kc_housing, target = "price")

lrn = LearnerRegrTorchTabnet$new()
lrn$param_set$values <- list(
  # verbose = TRUE,
  epochs = 50,
  lr_scheduler = "step",
  device = "cuda"
)

lrn_graph <- po("scale") %>>%
  po("learner", learner = lrn)

search_space <- ps(
  #penalty = p_dbl(lower = 0.0001, upper = 0.001),
  decision_width = p_int(lower = 8, upper = 64),
  attention_width = p_int(lower = 8, upper = 64),
  num_steps = p_int(lower = 3, upper = 10)
)

rsmp_holdout <- rsmp("holdout")
measure <- msr("regr.rmse")
trm_evals <- trm("evals", n_evals = 50)

instance <- TuningInstanceSingleCrit$new(
  task = task_regr,
  learner = lrn,
  resampling = rsmp_holdout,
  measure = measure,
  search_space = search_space,
  terminator = trm_evals
)

tuner <- tnr("random_search", batch_size = 5)

future::plan("multisession", workers = 12)

tuner$optimize(instance)

instance$result_learner_param_vals
