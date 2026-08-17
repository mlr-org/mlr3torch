with_torch_settings = function(seed, num_threads = 1, num_interop_threads = NULL, expr) {
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

  # Unlike the intraop count, this cannot be restored on exit: torch permits it to be set only once
  # per session. It is therefore only touched when the user asked for a specific value.
  if (!is.null(num_interop_threads) && num_interop_threads != torch_get_num_interop_threads()) {
    result = try(torch::torch_set_num_interop_threads(num_interop_threads), silent = TRUE)
    if (inherits(result, "try-error")) {
      stopf("Cannot set the number of interop threads to %s, it can only be set once per session and is already set to %s.", num_interop_threads, torch_get_num_interop_threads()) # nolint
    }
  }
  # sets the seed back when exiting the function
  if (!is.null(seed)) {
    local_torch_manual_seed(seed)
  }
  force(expr)
}
