library(tidytable)

# construct a task matching the pre-implemented tests
x_num = torch_randn(4, 3)
dt_num = setNames(as.data.table(as_array(x_num)), c("Num1", "Num2", "Num3"))
mat = matrix(nrow=4, ncol=2)
mat[1, ] = c(1L, 2L)
mat[2, ] = c(2L, 1L)
mat[3, ] = c(1L, 3L)
mat[4, ] = c(2L, 2L)
x_cat = torch_tensor(mat)
dt_cat = as.data.table(as_array(x_cat)) |>
  mutate(across(everything(), as.factor)) |>
  setNames(c("Cat1", "Cat2"))

y = factor(rbinom(n = 4, size = 1, prob = 0.5), levels = c(0, 1))

dt = bind_cols(y, dt_num, dt_cat) |>
  rename(y = ...1)
task = as_task_classif(dt, target = "y")

splits = partition(task)

d_embedding = 32