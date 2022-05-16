test_that("Can construc all paramsets and all parameters are covered", {
  for (act in torch_reflections$image_trafos) {
    param_set = paramsets_image_trafo$get(act)
    constructor = get_image_trafo(act)
    expected_ids = formalArgs(constructor) %??% character(0)
    expected_ids = setdiff(expected_ids, "img")
    ids = param_set$ids()
    expect_true(setequal(ids, expected_ids), info = act)
  }
})
