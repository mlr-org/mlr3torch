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

lrn_xgboost = as_learner(ppl("robustify") %>>% lrn("classif.xgboost"))
# TODO: factor, ordered feature types not allowed.
# TODO: throws an error at train time when required params are missing.
# Error in .__ParamSet__get_values(self = self, private = private, super = super,  : 
#   Missing required parameters: epochs, batch_size
# This happened PipeOp classif.ft_transformer's $train()
lrn_ft_transformer = as_learner(
  ppl("robustify") %>>% 
  lrn("classif.ft_transformer", epochs = 100, batch_size = 32)
)

learners = list(lrn_xgboost, lrn_ft_transformer)

rsmp_cv3 = rsmp("cv", folds = 3)

design = benchmark_grid(tasks, learners, rsmp_cv3)

future::plan("multisession")

time = bench::system_time(
  bmr <- benchmark(design)
)

fwrite(time, here("attic", "bmr_time.csv"))

print(as.data.table(bmr))
saveRDS(bmr, here("attic", "bmr.RDS"))