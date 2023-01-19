paramset_torchlearner = function() {
  ps(
    batch_size            = p_int(tags = c("train", "predict"), lower = 1L, default = 1L),
    epochs                = p_int(tags = c("train", "hotstart", "required"), lower = 0L),
    device                = p_fct(tags = c("train", "predict"), levels = c("auto", "cpu", "cuda"), default = "auto"),
    measures_train        = p_uty(tags = "train", custom_check = check_measures),
    measures_valid        = p_uty(tags = "train", custom_check = check_measures),
    augmentation          = p_uty(tags = "train"),
    callbacks             = p_uty(tags = "train", custom_check = check_callbacks),
    drop_last             = p_lgl(default = FALSE, tags = "train"),
    keep_last_prediction  = p_lgl(default = TRUE, tags = "train"),
    num_threads           = p_int(default = 1L, lower = 1L, tags = c("train", "predict", "threads")),
    shuffle               = p_lgl(default = TRUE, tags = "train"),
    early_stopping_rounds = p_int(default = 0L, tags = "train")
  )
}
