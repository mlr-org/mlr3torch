devtools::load_all()

dir = tempfile()
ci = col_info(tsk("tiny_imagenet")$backend$backend)
saveRDS(ci, "./data/col_info/tiny_imagenet.rds")
