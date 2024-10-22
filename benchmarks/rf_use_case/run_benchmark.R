library(mlr3)
library(mlr3learners)
library(mlr3oml)
library(mlr3torch)
library(mlr3tuning)
library(mlr3mbo)
library(bbotk)

library(bench)
library(data.table)
library(here)

options(mlr3oml.cache = here("benchmarks", "data", "oml"))

# define the tasks
cc18_small = fread(here(getOption("mlr3oml.cache"), "collections", "cc18_small.csv"))

task_list = mlr3misc::pmap(cc18_small, function(data_id, name, NumberOfFeatures, NumberOfInstances) tsk("oml", data_id = data_id))

mlp = lrn("classif.mlp",
  activation = nn_relu,
  neurons = to_tune(ps(
    n_layers = p_int(lower = 1, upper = 10), latent = p_int(10, 500),
    .extra_trafo = function(x, param_set) {
      list(neurons = rep(x$latent, x$n_layers))
    })
  ),
  batch_size = to_tune(c(64, 128, 256)),
  p = to_tune(0.1, 0.7),
  epochs = to_tune(lower = 1, upper = 500L, internal = TRUE),
  validate = "test",
  measures_valid = msr("classif.acc"),
  patience = 5,
  device = "cpu"
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
  measure = msr("classif.acc"),
  term_evals = 100
)

future::plan("multisession", workers = 8)

lrn_rf = lrn("classif.ranger")

options(mlr3.exec_random = FALSE)

design = benchmark_grid(
  task_list,
  learners = list(at, lrn_rf),
  resampling = rsmp("cv", folds = 3)
)
design = design[order(mlr3misc::ids(learner)), ]

time = bench::system_time(
  bmr <- benchmark(design)
)

bmrdt = as.data.table(bmr)

bmr$aggregate()[, .(task_id, learner_id, classif.ce)]

results_dir = here("benchmarks", "rf_use_case", "results")
if (!dir.exists(results_dir)) {
  dir.create(results_dir)
}
fwrite(bmr$aggregate()[, .(task_id, learner_id, classif.ce)], here(results_dir, "bmr_ce.csv"))
fwrite(as.data.table(as.list(time)), here(results_dir, "time.csv"))
