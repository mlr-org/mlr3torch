with_torch_settings = function(seed, num_threads, expr) {
  old_num_threads = torch_get_num_threads()
  on.exit({torch_set_num_threads(old_num_threads)},
    add = TRUE
  )

  torch_set_num_threads(num_threads)
  # sets the seed back when exiting the function
  local_torch_manual_seed(seed)

  withr::with_seed(
    seed = seed,
    code = {
      force(expr)
  })
}