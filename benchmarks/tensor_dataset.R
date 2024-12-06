devtools::load_all(".")
library(bench)

task = tsk("spam")$filter(1:2000)

learner1 = lrn("classif.mlp",
  epochs = 20L,
  batch_size = 256,
  device = "cpu"
)
learner2 = learner1$clone(deep = TRUE)
learner2$param_set$set_values(
  tensor_dataset = TRUE
)

f1 = function() learner1$train(task)
f2 = function() learner2$train(task)


x = mark(
  tensor_dataset = f2(),
  no_tensor_dataset = f1(),
  check = FALSE
)

print(x)

# 1 tensor_dataset     1.05m  1.05m    0.0158   41.28MB    1.64      1   104
# 2 no_tensor_dataset  1.52m  1.52m    0.0110    3.67GB    0.920     1    84

