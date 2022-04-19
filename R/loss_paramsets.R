make_paramset_loss = function(loss) {
  loss_paramsets$get(loss)
}

make_paramset_mse = function(loss) {
  ps(
    reduction = p_fct(levels = c("mean", "sum"), default = "mean", tags = "train")
  )
}

make_paramset_cross_entropy = function(loss) {
  ps(
    weight = p_uty(),
    ignore_index = p_int(),
    reduction = p_fct(levels = c("mean", "sum"), default = "mean")
  )
}

loss_paramsets = Dictionary$new()
loss_paramsets$add("mse", make_paramset_mse)
loss_paramsets$add("cross_entropy", make_paramset_cross_entropy)
