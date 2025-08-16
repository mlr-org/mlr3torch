library(here)

setwd(here("paper"))
source(here("paper", "benchmark", "benchmark.R"))

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

setup(
  here("paper", "benchmark", "registry-linux-gpu"),
  # This path is relative to the docker container, so no need to change it
 "/usr/bin/python3",
 here("paper")
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
saveRDS(result, here("paper", "benchmark", "result-linux-gpu.rds"))
