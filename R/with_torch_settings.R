with_torch_settings = function(seed, num_threads = 1, num_interop_threads = 1, expr) {
  old_num_threads = torch_get_num_threads()
  if (running_on_mac()) {
    if (!isTRUE(all.equal(num_threads, 1L))) {
      lg$warn("Cannot set number of threads on macOS.")
    }
  } else {
    on.exit({torch_set_num_threads(old_num_threads)},
      add = TRUE
    )
    torch_set_num_threads(num_threads)
  }

  if (num_interop_threads != torch_get_num_interop_threads()) {
    result = try(torch::torch_set_num_interop_threads(num_interop_threads), silent = TRUE)
    if (inherits(result, "try-error")) {
      lg$warn(sprintf("Can only set the interop threads once, keeping the previous value %s", torch_get_num_interop_threads()))
    }
  }
  # sets the seed back when exiting the function
  if (!is.null(seed)) {
    local_torch_manual_seed(seed)
  }
  force(expr)
}
