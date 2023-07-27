devtools::load_all()

dir = tempfile()
ci = col_info(get_private(tsk("tiny_imagenet")$backend)$.constructor())
saveRDS(ci, "./inst/col_info/tiny_imagenet.rds")
