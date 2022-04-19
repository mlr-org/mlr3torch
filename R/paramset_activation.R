make_paramset_activation = function(activation) {
  activation_paramseta$get(activation)
}

make_paramset_simple = function() {
  ps(
    inplace = p_lgl(default = FALSE, tags = "train")
  )
}

make_paramset_empty = function() {
  ps(
    inplace = p_lgl(default = FALSE, tags = "train")
  )
}


make_paramset_elu = function() {
  ps(
    alpha = p_dbl(tags = "train")
  )
}

make_paramset_hardhsrink = function() {
  ps()
}

make_paramset_harsigmoid = function() {
  ps()
}

make_paramset_leakyrelu = function() {
  ps()
}

make_paramset_logsigmoid = function() {
  ps()
}

make_paramset_multihead_attention = function() {
  ps()
}

make_paramset_prelu = function() {
  ps()
}

make_paramset_relu = function() {
  ps(
    inplace = p_lgl(default = FALSE, tags = "train")
  )
}

make_paramset_relu6 = function() {
  ps(
    inplace = p_lgl(default = FALSE, tags = "train")
  )
}

make_paramset_rrelu = function() {
  ps(
    inplace = p_lgl(default = FALSE, tags = "train")
  )
}

make_paramset_selu = function() {
  ps(
    inplace = p_lgl(tags = "train")
  )
}

make_paramset_celu = function() {
  ps(
    alpha = p_dbl(default = 1.0, tags = "train"),
    inplace = p_lgl(default = FALSE, tags = "train")
  )
}

make_paramset_gelu = function() {
  ps()
}

make_paramset_sigmoid = function() {
  ps()
}

make_paramset_silu = function() {
  ps(
    inplace = p_lgl(default = FALSE, tags = "train")
  )
}

make_paramset_mish = function() {
  ps(
    inplace = p_lgl(default = FALSE, tags = "train")
  )
}

make_paramset_softplus = function() {
  ps()
}

make_paramset_softshrink = function() {
  ps(
    lambda = p_dbl(default = 0.5, upper = 1, tags = "train")
  )
}

make_paramset_softsign = function() {
  ps()
}

make_paramset_tanh = function() {
  ps()
}

make_paramset_tanhshrink = function() {
  ps()
}

make_paramset_threshold = function() {
  ps(
    threshold = p_dbl(tags = "train"),
    value = p_dbl(tags = "train"),
    inplace = p_lgl(default = FALSE, tags = "train")

  )
}

make_paramset_glu = function() {
  ps(
    dim = p_int(default = -1L, lower = 1L, tags = "train", special_vals = list(-1L))
  )
}




activation_paramsets = Dictionary$new()
activation_paramsets$add("relu", make_paramset_simple)
activation_paramsets$add("sigmoid", make_paramset_empty)
      "elu", "hardshrink", "hardsigmoid", "hardtanh", "hardswish", "leakyrelu", "logsigmoid",
      "multihead_attention", "prelu", "relu", "relu6", "rrelu", "selu", "sigmoid", "silu", "mish",
      "softplus", "softshrink", "softsign", "tanh", "tanhshrink", "threshold", "glu"
