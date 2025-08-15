source("./paper/benchmark/benchmark.R")

#reg = loadRegistry("./paper/benchmark/registry", writeable = TRUE)
tbl = unwrap(getJobTable(findNotDone()))

tbl = tbl[device == "cpu" & repl == 1 & latent == 100L & !jit & n_layers == 16L, ]

print(tbl)

submitJobs(tbl$job.id)

source("./paper/benchmark/summarize.R")
