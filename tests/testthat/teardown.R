lg$set_threshold(old_threshold)
future::plan(old_plan)
lg_mlr3$set_threshold(old_threshold_mlr3)

assignInNamespace(ns = "mlr3torch", x = "auto_device",  value = prev_auto_device)
