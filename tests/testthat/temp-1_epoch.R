# maybe the train_loss isn't set yet

# maybe it's last_train_loss so at the end of the first epoch (or rather before the end)
# there is no last_train_loss

# when do

devtools::load_all()

cb = t_clbk("tb")

task = tsk("iris")

n_epochs = 1

mlp = lrn("classif.mlp",
          callbacks = cb,
          epochs = n_epochs, batch_size = 150, neurons = 10,
          validate = 0.2,
          measures_valid = msrs(c("classif.acc", "classif.ce")),
          measures_train = msrs(c("classif.acc", "classif.ce"))
)
mlp$param_set$set_values(cb.tb.path = tempfile())
mlp$param_set$set_values(cb.tb.log_train_loss = TRUE)

mlp$train(task)

events = mlr3misc::map(tfevents::collect_events(mlp$param_set$get_values()$cb.tb.path)$summary, unlist)

tensorflow::tensorboard(mlp$param_set$get_values()$cb.tb.path)

# with_logdir(mlp$param_set$get_values()$cb.tb.path,
#   log_event(manual_train_loss = mlp$param_set$get_values())
# )
# tfevents::log_event()
