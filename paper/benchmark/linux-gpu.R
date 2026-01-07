library(here)

source(here("benchmark", "benchmark.R"))

set.seed(44)

# Change this when not running this in the docker image
# Below is the correct python path for the CUDA docker image
PYTHON_PATH = "/usr/bin/python3"

if (!torch::cuda_is_available()) {
  stop("Cuda is not available for R-torch, please use the correct docker image.")
}

problem_design = expand.grid(
  list(
    n = N,
    p = P,
    epochs = EPOCHS,
    optimizer = c("sgd", "adamw"),
    batch_size = 32L,
    device = "cuda",
    n_layers = c(0L, 4L, 8L, 12L, 16L),
    latent = c(1000L, 3000L, 9000L)
  ),
  stringsAsFactors = FALSE
)

if (dir.exists(here("benchmark", "registry-linux-gpu"))) {
  stop("Registry benchmark/registry-linux-gpu already exists. Delete it to run the benchmark again.")
}

setup(
  here("benchmark", "registry-linux-gpu"),
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
  print(tbl[job.id == id, ])
  submitJobs(id)
  Sys.sleep(0.1)
}

source(here("benchmark", "summarize.R"))
result = summarize(tbl$job.id)
saveRDS(result, here("benchmark", "result-linux-gpu.rds"))
