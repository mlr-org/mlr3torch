# What a `"torch"` learner promises about its predictions, and what happens when a measure cannot
# be computed. Both used to fail silently; this script shows the current behaviour.
#
# Run from the package root:  Rscript example.R
# `TaskTorch` only exists on this branch, so this loads the source tree rather than an install.

pkgload::load_all(".", quiet = TRUE)
lgr::get_logger("mlr3")$set_threshold("warn")
options(warn = 1L)   # so the warnings below show up where they are raised

say = function(...) cat("\n== ", sprintf(...), "\n", sep = "")
refused = function(learner, predict_type) {
  cat(tryCatch({
    learner$predict_type = predict_type
    "accepted"
  }, error = function(e) trimws(conditionMessage(e))), "\n")
}

# ---------------------------------------------------------------------------
# Setup: a two-class problem, and two encoders that differ only in whether they
# produce probabilities.
# ---------------------------------------------------------------------------

d = withr::with_seed(1L, {
  d = data.frame(x1 = rnorm(60), x2 = rnorm(60))
  d$y = factor(ifelse(d$x1 > 0, "a", "b"))
  d
})

# response only -- a perfectly reasonable encoder for a problem where you only ever want a label
encode_response = function(task, network_output, predict_type) {
  p = as.matrix(network_output$cpu())
  lv = levels(task$truth())
  list(response = factor(lv[max.col(p)], levels = lv))
}

# the same thing, but able to hand out probabilities when asked
encode_prob = function(task, network_output, predict_type) {
  p = as.matrix(torch::nnf_softmax(network_output, dim = 2L)$cpu())
  colnames(p) = levels(task$truth())
  list(
    response = factor(colnames(p)[max.col(p)], levels = colnames(p)),
    prob = if (predict_type == "prob") p
  )
}

make_task = function(encoder, id, ...) {
  as_task_torch(d, target = "y", id = id,
    output_dim = function(task) nlevels(task$truth()),
    default_encoder = encoder, ...)
}

net = torch::nn_module("example_net",
  initialize = function(task) {
    self$linear = torch::nn_linear(task$n_features, output_dim_for(task))
  },
  forward = function(x) self$linear(x)
)

make_learner = function(...) {
  lrn("torch.module",
    module_generator = net,
    ingress_tokens = list(x = ingress_num()),
    loss = t_loss("cross_entropy"),
    target_batchgetter = function(data) {
      torch::torch_tensor(as.integer(data[[1L]]), dtype = torch::torch_long())
    },
    epochs = 1L, batch_size = 16L, ...
  )
}

msr_logloss = msr_torch("logloss",
  function(truth, prob) {
    p = pmin(pmax(prob[cbind(seq_along(truth), as.integer(truth))], 1e-7), 1 - 1e-7)
    -mean(log(p))
  },
  predict_type = "prob", range = c(0, Inf), minimize = TRUE
)

# ---------------------------------------------------------------------------
# 1. `prob` and `se` are opt-in.
#
# Whether a prediction can carry probabilities is decided by the task's
# `default_encoder`, and no task is in sight when the learner is built. A
# learner that claimed every predict type would accept `predict_type = "prob"`
# and then hand back a response-only prediction, without a word.
# ---------------------------------------------------------------------------

say("what a torch learner promises by default")
learner = make_learner()
print(learner$predict_types)          # response

say("asking for something it did not promise")
refused(learner, "prob")   # refused, at the point of the mistake

say("the same for se")
refused(learner, "se")

# ---------------------------------------------------------------------------
# 2. A learner built for a task whose encoder does produce probabilities says
#    so, and then everything downstream works.
# ---------------------------------------------------------------------------

task = make_task(encode_prob, "with_prob")
learner = make_learner(predict_types = c("response", "prob"))
learner$predict_type = "prob"
learner$train(task)
pred = learner$predict(task)

say("with prob declared")
print(pred$predict_types)             # response, prob
print(round(head(pred$prob, 3L), 3L))
print(pred$score(msr_logloss, task = task, learner = learner))

# The same has to be reachable through a Graph, or a GraphLearner could never predict `prob`.
say("through po(\"torch_model\")")
print(po("torch_model", predict_types = c("response", "prob"))$learner$predict_types)
print(po("torch_model")$learner$predict_types)

# ---------------------------------------------------------------------------
# 3. `classif` and `regr` are untouched: mlr3 knows what those tasks can do, so
#    their learners keep the defaults it expects.
# ---------------------------------------------------------------------------

say("the built-in task types keep their defaults")
print(lrn("classif.torch_featureless")$predict_types)   # response, prob
print(lrn("regr.torch_featureless")$predict_types)      # response

# ---------------------------------------------------------------------------
# 4. A measure that cannot be computed says why.
#
# Scoring during training must not end the run -- a network that diverges into
# NaN makes most measures assert, and the run may still recover -- so a failure
# still degrades to NaN. It is no longer silent about it: a measure that can
# never score this task used to leave a column of NaN and no explanation.
# ---------------------------------------------------------------------------

no_measure_task = make_task(encode_response, "no_default_measure")
learner = make_learner(measures_valid = msr("torch.default"), validate = 0.3)

say("validating against a measure the task cannot satisfy")
learner$train(no_measure_task)        # warns, naming the real cause, and carries on
print(learner$internal_valid_scores)  # NaN

# For contrast, a task that does carry a default measure scores normally.
ok_task = make_task(encode_response, "with_default_measure",
  default_measure = msr_torch("acc", function(truth, response) mean(truth == response),
    minimize = FALSE, range = c(0, 1)))
learner = make_learner(measures_valid = msr("torch.default"), validate = 0.3)

say("and one that can")
learner$train(ok_task)
print(learner$internal_valid_scores)
