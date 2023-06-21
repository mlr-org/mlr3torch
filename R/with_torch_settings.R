with_torch_settings = function(seed, num_threads, expr) {
  old_seed = get0(".Random.seed", globalenv(), mode = "integer", inherits = FALSE)
  old_num_threads = torch_get_num_threads()

  # We sample the torch seed (that we set after exiting from expr to avoid determinism after the function)
  # BEFORE we set the seed (otherwise it won't work)
  torch_seed = sample.int(10000000, 1)
  if (is.null(old_seed)) {
    print("No seed found")
    # In principle we don't need the runif(1) call below as long as we sample the torch_seed above
    # but we keep it so we don't forget adding it when the with_torch_manual_seed function is available
    runif(1)
    old_seed = get0(".Random.seed", globalenv(), mode = "integer", inherits = FALSE)
  }

  # FIXME: Ideally we want to set the torch manual seed back to its previous state but the function I would like to
  # use for that is only available in the dev version which I cannot install.
  # To ensure that we are reproducible but not everything is deterministic afterwards, we set the torch manual seed to
  # a random value after we are done. 
  # https://github.com/mlverse/torch/pull/999/files
  # https://github.com/mlverse/torch/issues/1052
  on.exit({
    # first sample, THEN reset the reset
    torch_manual_seed(torch_seed)
    assign(".Random.seed", old_seed, globalenv())
    torch_set_num_threads(old_num_threads)
    },
    add = TRUE
  )

  torch_set_num_threads(num_threads)
  set.seed(seed)
  torch_manual_seed(seed)

  force(expr)
}

with_seed = function(seed, expr) {
  old_seed = get0(".Random.seed", globalenv(), mode = "integer", inherits = FALSE)
  if (is.null(old_seed)) {
    runif(1L)
    old_seed = get0(".Random.seed", globalenv(), mode = "integer", inherits = FALSE)
  }

  on.exit(assign(".Random.seed", old_seed, globalenv()), add = TRUE)
  set.seed(seed)
  force(expr)
}
