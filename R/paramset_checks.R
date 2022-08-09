check_callbacks = function(x) {
  assert_true(all(map_lgl(x, function(x) inherits(x, "CallbackTorch"))))
}

check_measures = function(x) {
  if (!is.list(x)) {
    x = list(x)
  }
  if (test_list(x, types = "Measure")) {
    ids = map_chr(x, "id")
    if (test_names(ids, type = "unique")) {
      return(TRUE)
    }
  }

  "Parameter must be a Measure or list of Measures with valid ids."
}

check_network = function(x) {
  if (inherits(x, "nn_Module")) {
    "The network must be initialized by calling the function (and not with '$new()')."
  } else if (!test_class(x, "nn_module")) {
    "Must be a 'nn_module()'."
  } else {
    TRUE
  }
}
