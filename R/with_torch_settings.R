with_torch_settings = function(seed, num_threads = 1, expr) {
  old_num_threads = torch_get_num_threads()
  if (!running_on_mac()) {
    on.exit({torch_set_num_threads(old_num_threads)},
      add = TRUE
    )
    torch_set_num_threads(num_threads)
  } else {
    if (!isTRUE(all.equal(num_threads, 1L))) {
      lg$warn("Cannot set number of threads on macOS.")
    }
  }
  # sets the seed back when exiting the function
  if (!is.null(seed)) {
    local_torch_manual_seed(seed)
  }
  force(expr)
}
