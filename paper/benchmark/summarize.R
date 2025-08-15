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
  jt = getJobTable(ids) |> unwrap()
  jt = jt[, c("n_layers", "jit", "optimizer", "batch_size", "device", "opt_type", "algorithm", "repl", "tag", "latent", "epochs", "n", "p")]
  jt$time_total = get_result(ids, "time")
  jt$time_per_batch = jt$time_total / (ceiling(jt$n / jt$batch_size) * jt$epochs)
  jt$loss = get_result(ids, "loss")
  jt$memory = get_result(ids, "memory") / 2^30
  return(jt)
}
