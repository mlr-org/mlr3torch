
n_epochs = 10

task = tsk("iris")

mlp = lrn("classif.mlp",
          epochs = 10, batch_size = 150, neurons = c(100, 200, 300)
)
mlp$train(task)

names(mlp$network$parameters)

sela = select_all()
sela(names(mlp$network$parameters))

selg = select_grep("weight")
selg(names(mlp$network$parameters))

seln = select_name("0.weight")
seln(names(mlp$network$parameters))

seli = select_invert(select_name("0.weight"))
seli(names(mlp$network$parameters))

seln = select_none()
seln(names(mlp$network$parameters))

