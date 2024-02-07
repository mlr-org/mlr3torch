devtools::load_all()

ci = col_info(get_private(tsk("mnist")$backend)$.constructor())
saveRDS(ci, here::here("inst/col_info/mnist.rds"))
