make_paramset_loss = function(loss) {
  paramsets_loss$get(loss)
}

make_paramset_mse = function(loss) {
  ps(
    reduction = p_fct(levels = c("mean", "sum"), default = "mean", tags = "train")
  )
}

make_paramset_l1 = function(loss) {
  ps()
}

make_paramset_cross_entropy = function(loss) {
  ps(
    weight = p_uty(),
    ignore_index = p_int(),
    reduction = p_fct(levels = c("mean", "sum"), default = "mean")
  )
}

paramsets_loss = Dictionary$new()
paramsets_loss$add("mse", make_paramset_mse)
paramsets_loss$add("l1", make_paramset_mse)
paramsets_loss$add("cross_entropy", make_paramset_cross_entropy)
