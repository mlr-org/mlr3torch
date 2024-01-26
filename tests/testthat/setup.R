old_opts = options(
  warnPartialMatchArgs = TRUE,
  warnPartialMatchAttr = TRUE,
  warnPartialMatchDollar = TRUE
)

# https://github.com/HenrikBengtsson/Wishlist-for-R/issues/88
old_opts = lapply(old_opts, function(x) if (is.null(x)) FALSE else x)

lg_mlr3 = lgr::get_logger("mlr3")
old_threshold_mlr3 = lg_mlr3$threshold
lg_mlr3$set_threshold("warn")

old_threshold = lg$threshold
old_plan = future::plan()
lg$set_threshold("warn")
future::plan("sequential")

prev_auto_device = mlr3torch::auto_device

assignInNamespace(ns = "mlr3torch", x = "auto_device",  value = function(device = NULL) {
  if (device == "auto") {
    device = if (cuda_is_available()) {
      "cuda"
    } else if (backends_mps_is_available() && identical(Sys.getenv("GITHUB_ACTIONS"), "true")) {
      stop()
      # We are not the only ones experiencing issues:
      # https://discuss.pytorch.org/t/mps-back-end-out-of-memory-on-github-action/189773
      if (identical(Sys.getenv("GITHUB_ACTIONS"), "true") && nzchar(Sys.getenv("_R_CHECK_PACKAGE_NAME_", ""))) {
        "cpu"
      } else {
        "mps"
      }
    } else {
      "cpu"
    }
    lg$debug("Auto-detected device '%s'.", device)
  }
  return(device)
})
