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

task_list

# define the learners
# neurons = function(n_layers, latent_dim) {
#   rep(latent_dim, n_layers)
# }

# n_layers_values <- 1:5
# latent_dim_values <- seq(10, 200, by = 20)
# neurons_search_space <- mapply(
#   neurons,
#   expand.grid(n_layers = n_layers_values, latent_dim = latent_dim_values)$n_layers,
#   expand.grid(n_layers = n_layers_values, latent_dim = latent_dim_values)$latent_dim,
#   SIMPLIFY = FALSE
# )

mlp = lrn("classif.mlp",
  activation = nn_relu,
  neurons = to_tune(ps(
    n_layers = p_int(lower = 1, upper = 10), latent = p_int(10, 500),
    .extra_trafo = function(x, param_set) {
      list(neurons = rep(x$latent, x$n_layers))
    })
  ),
  # neurons = to_tune(neurons_search_space),
  batch_size = to_tune(c(64, 128, 256)),
  p = to_tune(0.1, 0.7),
  epochs = to_tune(upper = 1000L, internal = TRUE),
  validate = "test",
  measures_valid = msr("classif.acc"),
  patience = 10,
  device = "cpu"
)

mlp$encapsulate("callr", lrn("classif.featureless"))

# define an AutoTuner that wraps the classif.mlp
at = auto_tuner(
  learner = mlp,
  tuner = tnr("mbo"),
  resampling = rsmp("cv", folds = 5),
  measure = msr("classif.acc"),
  term_evals = 10
)

future::plan("multisession", workers = 8)

lrn_rf = lrn("classif.ranger")

options(mlr3.exec_random = FALSE)

design = benchmark_grid(
  task_list[[1]],
  learners = list(at, lrn_rf),
  resampling = rsmp("cv", folds = 3)
)
design = design[order(mlr3misc::ids(learner)), ]

time = bench::system_time(
  bmr <- benchmark(design)
)

bmrdt = as.data.table(bmr)

fwrite(bmrdt, here("R", "rf_use_case", "results", "bmrdt.csv"))
fwrite(time, here("R", "rf_use_case", "results", "time.csv"))