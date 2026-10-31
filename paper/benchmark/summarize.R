library(batchtools)
library(data.table)

get_result = function(ids, what) {
  if (is.null(ids)) ids = findDone()[[1]]
  sapply(ids, function(i) {
    res = loadResult(i)[[what]]
    if (is.null(res)) return(NA)
    res
  })
}

summarize = function(ids) {
  # Only jobs that finished without errors have results, e.g. when only a subset of the jobs was submitted.
  done = findDone(ids)$job.id
  if (!length(done)) stop("None of the jobs finished without errors.")
  if (length(done) < length(ids)) {
    warning(sprintf("Summarizing the %i of %i jobs that finished without errors.", length(done), length(ids)))
  }
  ids = done
  jt = getJobTable(ids) |> unwrap()
  if (is.null(jt$n_threads)) jt$n_threads = 1L
  jt = jt[, c("n_layers", "jit", "optimizer", "batch_size", "device", "opt_type", "algorithm", "repl", "latent", "epochs", "n", "p", "n_threads")]
  jt$time_total = get_result(ids, "time")
  jt$time_per_batch = jt$time_total / (ceiling(jt$n / jt$batch_size) * jt$epochs)
  jt$loss = get_result(ids, "loss")
  return(jt)
}
