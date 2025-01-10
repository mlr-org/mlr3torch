library(mlr3torch)
library(mlr3verse)
lgr::get_logger("mlr3")
set.seed(42)
torch_manual_seed(42)

# chunk 1
task = tsk("california_housing")
task

# chunk 2
preprocessing = po("encode", method = "one-hot") %>>%
  po("imputehist")

# chunk 3
ingress = po("torch_ingress_num")

# chunk 4
block = po("nn_linear", out_features = 32) %>>%
  ppl("branch", list(relu = po("nn_relu"), tanh = po("nn_tanh"))) %>>%
  po("nn_dropout")

# chunk 5
architecture = po("nn_block", block) %>>% po("nn_head")

# chunk 6
configuration = po("torch_loss", loss = t_loss("mse")) %>>%
  po("torch_optimizer", optimizer = t_opt("sgd"))

# chunk 7
model = po("torch_model_regr", device = "cpu", batch_size = 128)

# chunk 8
pipeline = preprocessing %>>%
  ingress %>>%
   architecture %>>%
  configuration %>>%
  model

learner = as_learner(pipeline)
learner$id = "custom_nn"


# chunk 9
learner$param_set$set_values(
  nn_block.nn_linear.out_features = to_tune(20, 200),
  nn_block.n_blocks = to_tune(1, 5),
  nn_block.branch.selection = to_tune(c("relu", "tanh")),
  nn_block.nn_dropout.p = to_tune(0.3, 0.7),
  torch_optimizer.lr = to_tune(10^-4, 10^-1, logscale = TRUE)
)

# chunk 10
set_validate(learner, "test")

# chunk 11
learner$param_set$set_values(
  torch_model_regr.patience = 10,
  torch_model_regr.measures_valid = msr("regr.mse"),
  torch_model_regr.epochs = to_tune(upper = 100, internal = TRUE)
)

# chunk 12
ti = tune(
  tuner = tnr("mbo"),
  resampling = rsmp("cv", folds = 3),
  measure = msr("internal_valid_score", minimize = TRUE),
  learner = learner,
  term_evals = 100,
  task = task
)
