library(here)

setwd(here("paper"))
source(here("paper", "benchmark", "benchmark.R"))

PYTHON_PATH = "/opt/venv/bin/python"

setup(
  here("paper", "benchmark", "registry-linux-cpu"),
  here("paper")
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

source(here("paper", "benchmark", "summarize.R"))
result = summarize(tbl$job.id)
saveRDS(result, here("paper", "benchmark", "result-linux-cpu.rds"))
