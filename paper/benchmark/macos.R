# start this from restored ./paper/envs/macos

library(here)

source(here("paper", "benchmark", "benchmark.R"))

# Adjust this path to the python executable that contains
# torch and numpy
PYTHON_PATH = "/Users/sebi/.venvs/mlr3torch-cpu/bin/python3"

problem_design = expand.grid(
  list(
    n = N,
    p = P,
    epochs = EPOCHS,
    # factor 10 smaller than cuda
    optimizer = c("sgd", "adamw"),
    batch_size = 32L,
    device = c("cpu", "mps"),
    n_layers = c(0L, 4L, 8L, 12L, 16L),
    latent = c(500L, 1000L, 2000L)
  ),
  stringsAsFactors = FALSE
)

setup(
  reg_path = here("paper", "benchmark", "registry-macos"),
  python_path = PYTHON_PATH,
  work_dir = here("paper", "envs", "macos")
)

addExperiments(
  prob.designs = list(
    runtime_train = problem_design
  ),
  algo.designs = list(
    rtorch = data.frame(
      jit = FALSE,
      opt_type = "ignite"
    ),
    mlr3torch = data.frame(
      jit = FALSE,
      opt_type = "ignite"
    ),
    pytorch = data.frame(
      jit = FALSE
    )
  ),
  repls = REPLS
)

tbl = unwrap(getJobTable())

for (id in sample(tbl$job.id)) {
  submitJobs(id)
  Sys.sleep(0.1)
}

source(here("paper", "benchmark", "summarize.R"))
result = summarize(tbl$job.id)
saveRDS(result, here("paper", "benchmark", "result-macos.rds"))
