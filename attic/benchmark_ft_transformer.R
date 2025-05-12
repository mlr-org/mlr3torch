devtools::load_all()

# library(mlr3oml)
library(mlr3learners)

library(here)

task_pima = tsk("pima")
task_sonar = tsk("sonar")
task_spam = tsk("spam")
tasks = list(task_pima, task_sonar, task_spam)
# task_mammographic_mass = as_task(otsk(id = 45557))
# tasks = list(task_breast_cancer, task_cali_housing, task_mammographic_mass)

glrn_xgboost = as_learner(po("imputesample") %>>% lrn("classif.xgboost"))
# TODO: factor, ordered feature types not allowed.
# TODO: throws an error at train time when required params are missing, maybe it would be better if this error was thrown earlier?
# Error in .__ParamSet__get_values(self = self, private = private, super = super,  :
#   Missing required parameters: epochs, batch_size
# This happened PipeOp classif.ft_transformer's $train()
# TODO: debug the behavior for defaults, doesn't really seem to work

# TODO: determine why wrapping the lrn() call in a function like this works
# make_ft_transformer_default = function(task_type, ...) {
#   params = list(
#      epochs = 1L,
#      batch_size = 32L,
#      n_blocks = 3L,
#      d_token = 32L,
#      query_idx = NULL
#   )
#   params = insert_named(params, list(...))
#   invoke(lrn, .key = sprintf("%s.ft_transformer", task_type), .args = params)
# }
# lrn_ft_transformer = make_ft_transformer_default("classif")

lrn_ft_transformer = lrn("classif.ft_transformer",
  attention_n_heads = 4,
  attention_dropout = 0.1,
  ffn_d_hidden = 100,
  ffn_dropout = 0.1,
  ffn_activation = nn_reglu,
  residual_dropout = 0.0,
  prenormalization = TRUE,
  is_first_layer = TRUE,
  attention_initialization = "kaiming",
  ffn_normalization = nn_layer_norm,
  attention_normalization = nn_layer_norm,
  query_idx = NULL,
  attention_bias = TRUE,
  ffn_bias_first = TRUE,
  ffn_bias_second = TRUE,
  # training
  epochs = 100L,
  batch_size = 32L,
  n_blocks = 3L,
  d_token = 32L
)
glrn_ft_transformer = as_learner(po("imputesample") %>>% lrn_ft_transformer)

learners = list(glrn_xgboost, glrn_ft_transformer)

rsmp_holdout = rsmp("holdout", ratio = 0.7)

design = benchmark_grid(tasks, learners, rsmp_holdout)

future::plan("multisession")

time = bench::system_time(
  bmr <- benchmark(design)
)

fwrite(time, here("attic", "bmr_time.csv"))

print(as.data.table(bmr))
saveRDS(bmr, here("attic", "bmr.RDS"))
