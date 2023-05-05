test_that("Can construc all paramsets and all parameters are covered", {
  for (act in mlr3torch_activations) {
    param_set = paramsets_activation$get(act)
    constructor = get_activation(act)
    expected_ids = formalArgs(constructor) %??% character(0)
    ids = param_set$ids()
    expect_true(setequal(ids, expected_ids), info = act)
  }
})
