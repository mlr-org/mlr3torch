devtools::load_all()

ci = col_info(get_private(tsk("tiny_imagenet")$backend)$.constructor())

saveRDS(ci, here::here("inst/col_info/tiny_imagenet.rds"))

mlr3:::DataBackendCbind$new(c)


split = factor(rep(c("train", "valid", "test"), times = c(100000, 10000, 10000)))

ci = rbind(ci, data.table(id = "split", type = "factor", levels = levels(split), label = NA, fix_factor_levels = TRUE))
setkeyv(ci)
