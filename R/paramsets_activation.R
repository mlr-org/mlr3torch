paramsets_activation = Dictionary$new()

make_paramset_elu = function() {
  ps(
    alpha = p_dbl(default = 1, tags = "train"),
    inplace = p_lgl(default = FALSE, tags = "train")
  )
}
paramsets_activation$add("elu", make_paramset_elu)

make_paramset_hardshrink = function() {
  ps(
    lambd = p_dbl(default = 0.5, tags = "train")
  )
}

paramsets_activation$add("hardshrink", make_paramset_hardshrink)

make_paramset_hardsigmoid = function() {
  ps()
}

paramsets_activation$add("hardsigmoid", make_paramset_hardsigmoid)

make_paramset_hardtanh = function() {
  ps(
    min_val = p_dbl(default = -1, tags = "train"),
    max_val = p_dbl(default = 1, tags = "train"),
    inplace = p_lgl(default = FALSE, tags = "train")
  )
}

paramsets_activation$add("hardtanh", make_paramset_hardtanh)

make_paramset_hardswish = function() {
  ps()
}

paramsets_activation$add("hardswish", make_paramset_hardswish)

make_paramset_leaky_relu = function() {
  ps(
    negative_slope = p_dbl(default = 0.01, tags = "train"),
    inplace = p_lgl(default = FALSE, tags = "train")
  )
}

paramsets_activation$add("leaky_relu", make_paramset_leaky_relu)

make_paramset_log_sigmoid = function() {
  ps()
}

paramsets_activation$add("log_sigmoid", make_paramset_log_sigmoid)


make_paramset_prelu = function() {
  ps(
    num_parameters = p_int(1, tags = "train"),
    init = p_dbl(default = 0.25, tags = "train")
  )
}

paramsets_activation$add("prelu", make_paramset_prelu)

make_paramset_relu = function() {
  ps(
    inplace = p_lgl(default = FALSE, tags = "train")
  )
}

paramsets_activation$add("relu", make_paramset_relu)

make_paramset_relu6 = function() {
  ps(
    inplace = p_lgl(default = FALSE, tags = "train")
  )
}

paramsets_activation$add("relu6", make_paramset_relu6)

make_paramset_rrelu = function() {
  ps(
    lower = p_dbl(default = 1 / 8, tags = "train"),
    upper = p_dbl(default = 1 / 3, tags = "train"),
    inplace = p_lgl(default = FALSE, tags = "train")
  )
}

paramsets_activation$add("rrelu", make_paramset_rrelu)

make_paramset_selu = function() {
  ps(
    inplace = p_lgl(tags = "train")
  )
}

paramsets_activation$add("selu", make_paramset_selu)

make_paramset_celu = function() {
  ps(
    alpha = p_dbl(default = 1.0, tags = "train"),
    inplace = p_lgl(default = FALSE, tags = "train")
  )
}

paramsets_activation$add("celu", make_paramset_celu)

make_paramset_gelu = function() {
  ps()
}

paramsets_activation$add("gelu", make_paramset_gelu)

make_paramset_sigmoid = function() {
  ps()
}

paramsets_activation$add("sigmoid", make_paramset_sigmoid)


make_paramset_softplus = function() {
  ps(
    beta = p_dbl(default = 1, tags = "train"),
    threshold = p_dbl(default = 20, tags = "train")
  )
}

paramsets_activation$add("softplus", make_paramset_softplus)

make_paramset_softshrink = function() {
  ps(
    lambd = p_dbl(default = 0.5, upper = 1, tags = "train")
  )
}

paramsets_activation$add("softshrink", make_paramset_softshrink)

make_paramset_softsign = function() {
  ps()
}

paramsets_activation$add("softsign", make_paramset_softsign)

make_paramset_tanh = function() {
  ps()
}

paramsets_activation$add("tanh", make_paramset_tanh)

make_paramset_tanhshrink = function() {
  ps()
}

paramsets_activation$add("tanhshrink", make_paramset_tanhshrink)

make_paramset_threshold = function() {
  ps(
    threshold = p_dbl(tags = "train"),
    value = p_dbl(tags = "train"),
    inplace = p_lgl(default = FALSE, tags = "train")

  )
}

paramsets_activation$add("threshold", make_paramset_threshold)

make_paramset_glu = function() {
  ps(
    dim = p_int(default = -1L, lower = 1L, tags = "train", special_vals = list(-1L))
  )
}

paramsets_activation$add("glu", make_paramset_glu)

# Only in Pyorch
# make_paramset_silu = function() {
#   ps(
#     inplace = p_lgl(default = FALSE, tags = "train")
#   )
# }
#
# paramsets_activation$add("silu", make_paramset_silu)

# make_paramset_mish = function() {
#   ps(
#     inplace = p_lgl(default = FALSE, tags = "train")
#   )
# }
#
# paramsets_activation$add("mish", make_paramset_mish)
