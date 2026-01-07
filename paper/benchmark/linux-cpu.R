library(here)

source(here("benchmark", "benchmark.R"))

set.seed(42)

# Change this when not running this in the docker image
# Below is the correct python path for the CPU docker image.
PYTHON_PATH = "/opt/venv/bin/python"

if (dir.exists(here("benchmark", "registry-linux-gpu"))) {
  stop("Registry benchmark/registry-linux-cpu already exists. Delete it to run the benchmark again.")
}

setup(
  here("benchmark", "registry-linux-cpu"),
  PYTHON_PATH,
  here()
)

problem_design = expand.grid(
  list(
    n = N,
    p = P,
    epochs = EPOCHS,
    optimizer = c("sgd", "adamw"),
    batch_size = 32L,
    device = "cpu",
    n_layers = c(0L, 4L, 8L, 12L, 16L),
    latent = c(100L, 200L, 400L)
  ),
  stringsAsFactors = FALSE
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

source(here("benchmark", "summarize.R"))
result = summarize(tbl$job.id)
saveRDS(result, here("benchmark", "result-linux-cpu.rds"))
