

cb = t_clbk("lr_scheduler", scheduler_fn = lr_step, step_size = 10)
cb = t_clbk("lr_scheduler", scheduler_fn = lr_cosine_annealing, ...)

task = tsk("iris")

mlp = lrn("classif.mlp",
          callbacks = cb,
          epochs = n_epochs, batch_size = 150, neurons = 10,
          measures_train = msrs(c("classif.acc", "classif.ce"))
)

mlp$train(task)