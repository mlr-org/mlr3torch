library(mlr3)
library(data.table)
library(mlr3torch)
library(paradox)

library(here)

# define the tasks
cc18_small = fread(here("data", "oml", "cc18_small.csv"))
cc18_small_datasets = mlr3misc::pmap(cc18_small, function(data_id, name, NumberOfFeatures, NumberOfInstances) {
  dt_name = here("data", "oml", paste0(name, "_", data_id, ".csv"))
  fread(dt_name)
})
# cc18_small_datasets

# cc18_small_datasets[[1]]

# TODO: determine whether we can use OML tasks "directly"
# didn't do this at first because they come with resamplings and we want to use our own resamplings
kc1_1067 = as_task_classif(cc18_small_datasets[[1]], target = "defects")
blood_1464 = as_task_classif(cc18_small_datasets[[2]], target = "Class")

tasks = list(kc1_1067, blood_1464)

# define the learners
mlp = lrn("classif.mlp",
  activation = nn_relu,
  neurons = to_tune(
    c(
      10, 20,
      c(10, 10), c(10, 20), c(20, 10), c(20, 20)
    )
  ),
  batch_size = to_tune(16, 32, 64),
  p = to_tune(0.1, 0.9),
  epochs = to_tune(upper = 1000L, internal = TRUE),
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

design = benchmark_grid(
  tasks,
  learners = list(at, lrn("classif.ranger"),
  resampling = rsmp("cv", folds = 10))
)

bmr = benchmark(design)

bmrdt = as.data.table(bmr)

fwrite(bmrdt, here("R", "rf_Use_case", "results", "bmrdt.csv"))

  # define an optimization strategy: grid search

  # define a search space: the parameters to tune over

    # neurons

    # batch size

    # dropout rate

    # epochs

  # use something standard (e.g. accuracy) as the tuning measure

  # use k-fold cross validation

  # set a number of evaluations for the tuner

# TODO: set up the tuning space for the neurons and layers

# layers_search_space <- 1:5
# neurons_search_space <- seq(10, 50, by = 10)

# generate_permutations <- function(layers_search_space, neurons_search_space) {
#   result <- list()
  
#   for (layers in layers_search_space) {
#     # Generate all permutations with replacement
#     perms <- expand.grid(replicate(layers, list(neurons_search_space), simplify = FALSE))
    
#     # Convert each row to a vector and add to the result
#     result <- c(result, apply(perms, 1, as.numeric))
#   }
  
#   return(result)
# }

# permutations <- generate_permutations(layers_search_space, neurons_search_space)

# head(permutations)
