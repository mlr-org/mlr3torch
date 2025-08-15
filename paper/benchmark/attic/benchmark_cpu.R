library(batchtools)
library(mlr3misc)

reg = makeExperimentRegistry(
  file.dir = here::here("paper", "benchmark", "registry_cpu"),
  packages = "checkmate"
)
reg$cluster.functions = makeClusterFunctionsInteractive()

source(here::here("paper/benchmark/time_rtorch.R"))

batchExport(list(
  time_rtorch = time_rtorch
))

# The algorithm should return the total runtime needed for training, the SD, but also the performance of the training losses so we know it is all working
addProblem("runtime_train",
  data = NULL,
  fun = function(epochs, batch_size, n_layers, latent, n, p, optimizer, device, ...) {
    problem = list(
      epochs = assert_int(epochs),
      batch_size = assert_int(batch_size),
      n_layers = assert_int(n_layers),
      latent = assert_int(latent),
      n = assert_int(n),
      p = assert_int(p),
      optimizer = assert_choice(optimizer, c("ignite_adamw", "adamw", "sgd", "ignite_sgd")),
      device = assert_choice(device, c("cuda", "cpu"))
    )

    problem
  }
)

# pytorch needs to be submitted with an active pytorch environment
addAlgorithm("pytorch",
  fun = function(instance, job, data, jit, ...) {
    f = function(...) {
      library(reticulate)
      x = try({
        reticulate::use_python("/usr/bin/python3", required = TRUE)
        reticulate::source_python(here::here("paper/benchmark/time_pytorch.py"))
        print(reticulate::py_config())
        time_pytorch(...)
      }, silent = TRUE)
      print(x)

    }
    args = c(instance, list(seed = job$seed, jit = jit))
    #do.call(f, args)
    callr::r(f, args = args)
  }
)

addAlgorithm("rtorch",
  fun = function(instance, job, opt_type, jit,...) {
    assert_choice(opt_type, c("standard", "ignite"))
    if (opt_type == "ignite") {
      instance$optimizer = paste0("ignite_", instance$optimizer)
    }
    #do.call(time_rtorch, args = c(instance, list(seed = job$seed, jit = jit)))
    callr::r(time_rtorch, args = c(instance, list(seed = job$seed, jit = jit)))
  }
)

addAlgorithm("mlr3torch",
  fun = function(instance, job, opt_type, jit, ...) {
    if (opt_type == "ignite") {
      instance$optimizer = paste0("ignite_", instance$optimizer)
    }
    do.call(time_rtorch, args = c(instance, list(seed = job$seed, mlr3torch = TRUE, jit = jit)))
    #callr::r(time_rtorch, args = c(instance, list(seed = job$seed, mlr3torch = TRUE, jit = jit)))
  }
)

# global config:
REPLS = 4L
EPOCHS = 20L
N = 2000L
P = 1000L

# cuda experiments:


problem_design = expand.grid(list(
  n          = N,
  p          = P,
  epochs = EPOCHS,
  latent = c(1000, 2500, 5000),
  optimizer = c("sgd", "adamw"),
  batch_size = 32L,
  device     = "cuda",
  n_layers = c(2L, 4L, 6L, 8L, 10L, 12L, 14L, 16L)
), stringsAsFactors = FALSE)


addExperiments(
  prob.designs = list(
    runtime_train = problem_design
  ),
  algo.designs = list(
    rtorch = data.frame(
      jit = c(FALSE, TRUE),
      opt_type = c("ignite"),
      tag = "cuda_exp"
    ),
    mlr3torch = data.frame(
      jit = c(FALSE, TRUE),
      opt_type = c("ignite"),
      tag = "cuda_exp"
    ),
    pytorch = data.frame(
      jit = c(FALSE, TRUE),
      tag = "cuda_exp"
    )
  ),
  repls = REPLS
)

# cpu experiments:
# (need smaller networks, otherwise too expensive with the cuda config)

problem_design = expand.grid(list(
  n          = N,
  p          = P,
  epochs = EPOCHS,
  # factor 10 smaller than cuda
  latent = c(50, 100, 200),
  optimizer = c("sgd", "adamw"),
  batch_size = 32L,
  device     = "cpu",
  n_layers = c(2L, 4L, 6L, 8L, 10L, 12L, 14L, 16L)
), stringsAsFactors = FALSE)

addExperiments(
  prob.designs = list(
    runtime_train = problem_design
  ),
  algo.designs = list(
    rtorch = data.frame(
      jit = c(FALSE, TRUE),
      opt_type = c("ignite"),
      tag = "cpu_exp"
    ),
    mlr3torch = data.frame(
      jit = c(FALSE, TRUE),
      opt_type = c("ignite"),
      tag = "cpu_exp"
    ),
    pytorch = data.frame(
      jit = c(FALSE, TRUE),
      tag = "cpu_exp"
    )
  ),
  repls = REPLS
)

ids = sample(findJobs()[[1L]])
tbl = unwrap(getJobTable())
ids = tbl[device == "cpu" & n_layers == 10 & latent == 200 & jit & optimizer == "adamw" & repl == 1, ]$job.id
# there is a bug in batchtools that sorts the IDs
# when submitting them together
for (id in sample(ids)) {
  submitJobs(id)
  Sys.sleep(0.1)
}
