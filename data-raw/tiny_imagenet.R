devtools::load_all()

ci = col_info(get_private(tsk("tiny_imagenet")$backend)$.constructor())

saveRDS(ci, here::here("inst/col_info/tiny_imagenet.rds"))