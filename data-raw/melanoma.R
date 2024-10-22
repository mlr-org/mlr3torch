devtools::load_all()

ci = col_info(get_private(tsk("melanoma")$backend)$.constructor())

saveRDS(ci, here::here("inst/col_ino/melanoma.rds"))

mlr3::DataBackendCbind$new(c)

# split

# ci