criterion_paramsets = Dictionary$new()

make_paramset_criterion = function(criterion) {
  criterion_paramsets$get(criterion)
}

make_paramset_cross_entropy = function() {
  ps(
    reduction = p_fct(default = "mean", levels = c("mean", "sum"))
  )
}

criterion_paramsets$add("cross_entropy", make_paramset_cross_entropy)
