devtools::load_all()

# manually construct the task once

ci = col_info(get_private(tsk("melanoma")$backend)$.constructor())

saveRDS(ci, here::here("inst/col_ino/melanoma.rds"))

mlr3::DataBackendCbind$new(c)

# split

# ci