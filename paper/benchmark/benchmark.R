library(batchtools)
library(mlr3misc)

# Environment for the benchmark subprocesses.
# torch_set_num_threads() only applies to the calling thread, but (R) torch runs the backward pass in a
# separate thread, which otherwise uses all available cores. Setting OMP_NUM_THREADS also limits these threads.
thread_env = function(n_threads) {
  n_threads = as.character(n_threads)
  c(callr::rcmd_safe_env(), OMP_NUM_THREADS = n_threads, MKL_NUM_THREADS = n_threads)
}

setup = function(reg_path, python_path, work_dir) {

  print_setup_info(reg_path, python_path, work_dir)
  
  
  if (file.exists(reg_path)) {
    msg <- sprintf("Registry already exists at path %s. Delete the folder it to run the benchmark again.", reg_path)
    if (!interactive()) {
      stop(msg)
    }
    answer <- readline(sprintf("Registry already exists at path %s. Delete it to run the benchmark again? (y/n)", reg_path))
     if (answer == "y") {
       unlink(reg_path, recursive = TRUE)
     } else {
       stop(msg)
     }
  }

  reg = makeExperimentRegistry(
    file.dir = reg_path,
    work.dir = work_dir,
    packages = "checkmate",
    seed = 123
  )
  reg$cluster.functions = makeClusterFunctionsInteractive()

  source(here::here("benchmark", "time_rtorch.R"))

  batchExport(list(
    time_rtorch = time_rtorch, # nolint
    thread_env = thread_env
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
      n_threads = 1L,
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
        device = assert_choice(device, c("cuda", "cpu", "mps")),
        n_threads = assert_int(n_threads, lower = 1L)
      )

      problem
    }
  )

  addAlgorithm("pytorch", fun = function(instance, job, data, jit, ...) {
    print(instance)
    f = function(..., python_path) {
      library(reticulate)
      x = try(
        {
          #reticulate::use_python("/opt/homebrew/Caskroom/mambaforge/base/bin/python3", required = TRUE)
          reticulate::use_python(python_path, required = TRUE)
          reticulate::source_python(here::here("benchmark", "time_pytorch.py"))
          print(reticulate::py_config())
          time_pytorch(...) # nolint
        },
        silent = TRUE
      )
      print(x)
    }
    args = c(instance, list(seed = job$seed, jit = jit, python_path = python_path))
    #do.call(f, args)
    callr::r(f, args = args, env = thread_env(instance$n_threads))
  })

  addAlgorithm("rtorch", fun = function(instance, job, opt_type, jit, ...) {
    print(instance)
    assert_choice(opt_type, c("standard", "ignite"))
    if (opt_type == "ignite") {
      instance$optimizer = paste0("ignite_", instance$optimizer)
    }
    #do.call(time_rtorch, args = c(instance, list(seed = job$seed, jit = jit))) # nolint
    callr::r(time_rtorch, args = c(instance, list(seed = job$seed, jit = jit)), env = thread_env(instance$n_threads)) # nolint
  })

  addAlgorithm("mlr3torch", fun = function(instance, job, opt_type, jit, ...) {
    print(instance)
    if (opt_type == "ignite") {
      instance$optimizer = paste0("ignite_", instance$optimizer)
    }
    callr::r(
      time_rtorch, # nolint
      args = c(instance, list(seed = job$seed, mlr3torch = TRUE, jit = jit)),
      env = thread_env(instance$n_threads)
    )
    #do.call(time_rtorch, args = c(instance, list(seed = job$seed, mlr3torch = TRUE, jit = jit)))
  })
}

# global config:
REPLS = 10L
EPOCHS = 20L
N = 2000L
P = 1000L

print_setup_info = function(reg_path, python_path, work_dir) {
  cat("Session Info:\n")
  print(sessionInfo())
  cat("Library Paths:\n")
  for (path in .libPaths()) {
    cat("  -", path, "\n")
  }
  cat("Working Directory:", getwd(), "\n")

  cat("Subfolders of working directory:\n")
  for (folder in list.files(work_dir)) {
    cat("  -", folder, "\n")
  }

  # Function arguments
  cat("--- FUNCTION ARGUMENTS ---\n")
  cat("  Registry Path:", reg_path, "\n")
  cat("  Python Path:", python_path, " (", if (file.exists(python_path)) "exists" else "does not exist", ")\n")
  cat("  Work Directory:", work_dir, "\n\n")
  cat("Cuda is available:", torch::cuda_is_available(), "\n")
  out <- try(callr::r(function(python_path) {
    reticulate::use_python(python_path, required = TRUE)
    return(reticulate::py_config())
  }, show = TRUE, args = list(python_path = python_path)), silent = TRUE)
  if (inherits(out, "try-error")) {
    cat("Error occurred while calling Python:\n")
    print(out)
  } else {
    cat("Python configuration:\n")
    print(out)
  }
}
