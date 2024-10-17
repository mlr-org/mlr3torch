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

# when working on the GPU server, don't forget to activate the mamba environment with the torch installation

# define the tasks
cc18_small = fread(here(getOption("mlr3oml.cache"), "collections", "cc18_small.csv"))

task_list = mlr3misc::pmap(cc18_small, function(data_id, name, NumberOfFeatures, NumberOfInstances) tsk("oml", data_id = data_id))

task_list

# define the learners
neurons = function(n_layers, latent_dim) {
  rep(latent_dim, n_layers)
}

n_layers_values <- 1:10
latent_dim_values <- seq(10, 500, by = 10)
neurons_search_space <- mapply(
  neurons,
  expand.grid(n_layers = n_layers_values, latent_dim = latent_dim_values)$n_layers,
  expand.grid(n_layers = n_layers_values, latent_dim = latent_dim_values)$latent_dim,
  SIMPLIFY = FALSE
)

mlp = lrn("classif.mlp",
  activation = nn_relu,
  # neurons = to_tune(ps(
  #   n_layers = p_int(lower = 1, upper = 10), latent = p_int(10, 500),
  #   .extra_trafo = function(x, param_set) {
  #     list(neurons = rep(x$latent, x$n_layers))
  #   })
  # ),
  neurons = to_tune(neurons_search_space),
  batch_size = to_tune(c(16, 32, 64, 128, 256)),
  p = to_tune(0.1, 0.9),
  epochs = to_tune(upper = 1000L, internal = TRUE),
  validate = 0.3,
  measures_valid = msr("classif.acc"),
  patience = 10,
  device = "cpu"
)

# define the optimization strategy
bayesopt_ego = mlr_loop_functions$get("bayesopt_ego")
surrogate = srlrn(lrn("regr.km", covtype = "matern5_2",
  optim.method = "BFGS", control = list(trace = FALSE)))
acq_function = acqf("ei")
acq_optimizer = acqo(opt("nloptr", algorithm = "NLOPT_GN_ORIG_DIRECT"),
  terminator = trm("stagnation", iters = 100, threshold = 1e-5))

tnr_mbo = tnr("mbo",
    loop_function = bayesopt_ego,
    surrogate = surrogate,
    acq_function = acq_function,
    acq_optimizer = acq_optimizer)

# define an AutoTuner that wraps the classif.mlp
at = auto_tuner(
  learner = mlp,
  tuner = tnr_mbo,
  resampling = rsmp("cv"),
  measure = msr("classif.acc"),
  term_evals = 1000
)

future::plan("multisession", workers = 8)

lrn_rf = lrn("classif.ranger")
design = benchmark_grid(
  task_list,
  learners = list(at, lrn_rf),
  resampling = rsmp("cv", folds = 10)
)

time = bench::system_time(
  bmr <- benchmark(design)
)

bmrdt = as.data.table(bmr)

fwrite(bmrdt, here("R", "rf_use_case", "results", "bmrdt.csv"))
fwrite(time, here("R", "rf_use_case", "results", "time.csv"))