library(batchtools)
library(mlr3misc)

PYTHON_PATH = "/opt/venv/bin/python"

setup = function(reg_path, work_dir) {
  reg = makeExperimentRegistry(
    file.dir = reg_path,
    work.dir = work_dir,
    packages = "checkmate"
  )
  reg$cluster.functions = makeClusterFunctionsInteractive()

  source(here::here("paper/benchmark/time_rtorch.R"))

  batchExport(list(
    time_rtorch = time_rtorch # nolint
  ))

  addProblem(
    "runtime_train",
    data = NULL,
    fun = function(
      epochs,
      batch_size,
      n_layers,
      latent,
      n,
      p,
      optimizer,
      device,
      ...
    ) {
      problem = list(
        epochs = assert_int(epochs),
        batch_size = assert_int(batch_size),
        n_layers = assert_int(n_layers),
        latent = assert_int(latent),
        n = assert_int(n),
        p = assert_int(p),
        optimizer = assert_choice(
          optimizer,
          c("ignite_adamw", "adamw", "sgd", "ignite_sgd")
        ),
        device = assert_choice(device, c("cuda", "cpu", "mps"))
      )

      problem
    }
  )

  addAlgorithm("pytorch", fun = function(instance, job, data, jit, ...) {
    f = function(..., python_path) {
      library(reticulate)
      x = try(
        {
          #reticulate::use_python("/opt/homebrew/Caskroom/mambaforge/base/bin/python3", required = TRUE)
          reticulate::use_python(python_path, required = TRUE)
          reticulate::source_python(here::here("paper/benchmark/time_pytorch.py"))
          print(reticulate::py_config())
          time_pytorch(...) # nolint
        },
        silent = TRUE
      )
      print(x)
    }
    args = c(instance, list(seed = job$seed, jit = jit, python_path = PYTHON_PATH))
    #do.call(f, args)
    callr::r(f, args = args)
  })

  addAlgorithm("rtorch", fun = function(instance, job, opt_type, jit, ...) {
    assert_choice(opt_type, c("standard", "ignite"))
    if (opt_type == "ignite") {
      instance$optimizer = paste0("ignite_", instance$optimizer)
    }
    #do.call(time_rtorch, args = c(instance, list(seed = job$seed, jit = jit))) # nolint
    callr::r(time_rtorch, args = c(instance, list(seed = job$seed, jit = jit))) # nolint
  })

  addAlgorithm("mlr3torch", fun = function(instance, job, opt_type, jit, ...) {
    if (opt_type == "ignite") {
      instance$optimizer = paste0("ignite_", instance$optimizer)
    }
    callr::r(
      time_rtorch, # nolint
      args = c(instance, list(seed = job$seed, mlr3torch = TRUE, jit = jit))
    )
    #do.call(time_rtorch, args = c(instance, list(seed = job$seed, mlr3torch = TRUE, jit = jit)))
  })
}

# global config:
REPLS = 10L
EPOCHS = 20L
N = 2000L
P = 1000L
