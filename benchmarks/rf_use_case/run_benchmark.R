library(mlr3)
library(mlr3learners)
library(mlr3oml)
library(mlr3torch)
library(mlr3tuning)

library(paradox)

library(data.table)

library(here)

options(mlr3oml.cache = here("benchmarks", "data", "oml"))

# define the tasks
cc18_small = fread(here(getOption("mlr3oml.cache"), "collections", "cc18_small.csv"))

task_list = mlr3misc::pmap(cc18_small, function(data_id, name, NumberOfFeatures, NumberOfInstances) tsk("oml", data_id = data_id))

task_list

# define the learners
mlp = lrn("classif.mlp",
  activation = nn_relu,
  neurons = to_tune(ps(
    n_layers = p_int(lower = 1, upper = 10), latent = p_int(10, 500),
    .extra_trafo = function(x, param_set) {
      list(neurons = rep(x$latent, x$n_layers))
    })
  ),
  batch_size = to_tune(c(16, 32, 64)),
  p = to_tune(0.1, 0.9),
  epochs = to_tune(upper = 100, internal = TRUE),
  validate = 0.3,
  measures_valid = msr("classif.acc"),
  patience = 10,
  device = "cpu"
)

# define an AutoTuner that wraps the classif.mlp
at = auto_tuner(
  learner = mlp,
  tuner = tnr("grid_search"),
  resampling = rsmp("cv"),
  measure = msr("classif.acc"),
  term_evals = 10
)

future::plan("multisession")

lrn_rf = lrn("classif.ranger")
design = benchmark_grid(
  task_list,
  learners = list(at, lrn_rf),
  resampling = rsmp("cv", folds = 10))

bench::system_time(
  bmr <- benchmark(design)
)

bmrdt = as.data.table(bmr)

fwrite(bmrdt, here("R", "rf_use_case", "results", "bmrdt.csv"))
