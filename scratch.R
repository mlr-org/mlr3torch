devtools::load_all("~/mlr/mlr3torch")

obj = pot("linear")
obj


x = pot("ingress_num")$train(list(tsk("iris")))

y = pot("linear", out_features = 10L)$train(x)

md = y$output
unclass(md)
