

inferps = function(fn, ignore = character(0)) {
  assert_function(fn)
  assert_character(ignore, any.missing = FALSE)
  frm = formals(fn)
  frm = frm[names(frm) %nin% ignore]

  frm_domains = lapply(frm, function(formal) {
    switch(typeof(formal),
      logical = p_lgl(),
      integer = p_int(),
      double = p_dbl(),
      p_uty()
    )
  })

  do.call(paradox::ps, frm_domains)
}

check_callbacks = function(x) {
  check_list(x, types = "R6ClassGenerator", any.missing = FALSE)
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
