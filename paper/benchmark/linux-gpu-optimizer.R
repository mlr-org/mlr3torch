library(here)

source(here("benchmark", "benchmark.R"))

# Change this when not running this in the docker image
# Below is the correct python path for the CUDA docker image
PYTHON_PATH = "/usr/bin/python3"

problem_design = expand.grid(
  list(
    n = N,
    p = P,
    epochs = EPOCHS,
    optimizer = c("sgd", "adamw"),
    batch_size = 32L,
    device = "cuda",
    n_layers = c(0L, 4L, 8L, 12L, 16L),
    latent = c(1000L)
  ),
  stringsAsFactors = FALSE
)

if (dir.exists(here("benchmark", "registry-linux-gpu-optimizer"))) {
  stop("Registry already exists. Delete it to run the benchmark again.")
}

setup(
  here("benchmark", "registry-linux-gpu-optimizer"),
  PYTHON_PATH,
  here()
)

addExperiments(
  prob.designs = list(
    runtime_train = problem_design
  ),
  algo.designs = list(
    rtorch = data.frame(
      jit = FALSE,
      opt_type = c("ignite", "standard")
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
saveRDS(result, here("benchmark", "result-linux-gpu-optimizer.rds"))
