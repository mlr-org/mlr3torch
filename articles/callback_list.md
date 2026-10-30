# Callbacks

Below is a list of the predefined callbacks that are available in
`mlr3torch`. `Weight` is the order in which a callback is called within
a stage: a callback with a higher weight is called after one with a
lower weight, see the section *Ordering* of `CallbackSet`.

| Key | Label | Weight | Packages |
|:---|:---|---:|:---|
| [checkpoint](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.checkpoint.html) | Checkpoint | Inf | torch |
| [history](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.history.html) | History | 0 | torch |
| [lr_cosine_annealing](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.lr_scheduler.html) | Cosine Annealing LR Scheduler | 0 | torch |
| [lr_lambda](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.lr_scheduler.html) | Multiplication by Function LR Scheduler | 0 | torch |
| [lr_multiplicative](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.lr_scheduler.html) | Multiplication by Factor LR Scheduler | 0 | torch |
| [lr_one_cycle](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.lr_scheduler_one_cycle.html) | 1cycle LR Scheduler | 0 | torch |
| [lr_reduce_on_plateau](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.lr_scheduler_reduce_on_plateau.html) | Reduce on Plateau LR Scheduler | 0 | torch |
| [lr_step](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.lr_scheduler.html) | Step Decay LR Scheduler | 0 | torch |
| [progress](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.progress.html) | Progress | 0 | progress, torch |
| [tb](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.tb.html) | TensorBoard | 0 | tfevents, torch |
| [unfreeze](https://mlr3torch.mlr-org.com/reference/mlr_callback_set.unfreeze.html) | Unfreeze | 0 | torch |
