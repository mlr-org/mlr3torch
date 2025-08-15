source("./paper/benchmark/benchmark.R")

#reg = loadRegistry("./paper/benchmark/registry", writeable = TRUE)
tbl = unwrap(getJobTable(findNotDone()))

tbl = tbl[device == "cpu" & repl == 1 & latent == 100L & jit & algorithm == "rtorch", ]

print(tbl)

submitJobs(tbl$job.id)

source("./paper/benchmark/summarize.R")
