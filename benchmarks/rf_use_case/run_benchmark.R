library(mlr3verse)
library(mlr3oml)
library(mlr3torch)
library(mlr3batchmark)
library(mlr3mbo)
library(mlr3tuning)

ids = c(1067, 1464, 1485, 1494, 40994)
task_list = lapply(ids, function(id) tsk("oml", data_id = id))

mlp = lrn("classif.mlp",
  activation = nn_relu,
  n_layers = to_tune(lower = 1, upper = 10),
  neurons = to_tune(p_int(lower = 10, upper = 1000)),
  batch_size = to_tune(c(64, 128, 256)),
  p = to_tune(0.1, 0.9),
  epochs = to_tune(lower = 1, upper = 1000L, internal = TRUE),
  validate = "test",
  measures_valid = msr("classif.logloss"),
  patience = 10,
  device = "auto",
  predict_type = "prob"
)

mlp$encapsulate("callr", lrn("classif.featureless"))

surrogate = srlrn(as_learner(po("imputesample", affect_columns = selector_type("logical")) %>>%
  po("imputeoor", multiplier = 3, affect_columns = selector_type(c("integer", "numeric", "character", "factor", "ordered"))) %>>%
  po("colapply", applicator = as.factor, affect_columns = selector_type("character")) %>>%
  lrn("regr.ranger")), catch_errors = TRUE)

# define an AutoTuner that wraps the classif.mlp
at = auto_tuner(
  learner = mlp,
  tuner = tnr("mbo", surrogate = surrogate),
  resampling = rsmp("cv", folds = 5),
  measure = msr("internal_valid_score", minimize = TRUE),
  term_evals = 1
)

lrn_rf = lrn("classif.ranger")

design = benchmark_grid(
  task_list,
  learners = list(at, lrn_rf),
  resampling = rsmp("cv", folds = 3)
)

design1 = benchmark_grid(
  task_list[[1]],
  learners = list(at, lrn_rf),
  resampling = rsmp("holdout")
)

benchmark(design1)

reg = makeExperimentRegistry(
  file.dir = here("benchmarks", "rf_use_case", "reg"),
  packages = c("mlr3verse", "mlr3oml", "mlr3torch", "batchmark")
)

batchmark(design)
