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

# Set attention_width to NULL and tune over decision_width, as model authors
# recommend N_d == N_a and {tabnet} sets N_d == N_a if either is NULL
lrn$param_set$values <- list(
  # verbose = TRUE,
  epochs = 50,
  lr_scheduler = "step",
  device = "cuda",
  attention_width = NULL
)

lrn_graph <- po("scale") %>>%
  po("learner", learner = lrn)

search_space <- ps(
  #penalty = p_dbl(lower = 0.0001, upper = 0.001),
  decision_width = p_int(lower = log2(8), upper = log2(64), trafo = function(x) 2^x),
  #attention_width = p_int(lower = 8, upper = 64),
  num_steps = p_int(lower = 3, upper = 10)
)

instance <- TuningInstanceSingleCrit$new(
  task = task_regr,
  learner = lrn,
  resampling = rsmp("holdout"),
  measure = msr("regr.rmse"),
  search_space = search_space,
  terminator = trm("evals", n_evals = 50)
)

tuner <- tnr("grid_search", resolution = 1, batch_size = 4)

future::plan("multisession", workers = 4)

tuner$optimize(instance)

instance$result_learner_param_vals

saveRDS(instance, "attic/tuning-result.rds")
