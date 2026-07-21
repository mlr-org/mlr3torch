test_that("nn_tab_transformer blocks work", {
  x = torch_randn(3, 4, 8)

  attn = nn_tab_transformer_attention(d_token = 8, n_heads = 2, dim_head = 4)
  expect_class(attn, "nn_tab_transformer_attention")
  expect_equal(attn(x)$shape, c(3, 4, 8))

  ffn = nn_tab_transformer_ffn(d_token = 8)
  expect_class(ffn, "nn_tab_transformer_ffn")
  expect_equal(ffn(x)$shape, c(3, 4, 8))

  block = nn_tab_transformer_block(d_token = 8, n_heads = 2, dim_head = 4, attn_dropout = 0,
    ff_dropout = 0)
  expect_class(block, "nn_tab_transformer_block")
  expect_equal(block(x)$shape, c(3, 4, 8))
})

test_that("nn_tab_transformer forward shapes are correct", {
  cardinalities = c(3, 4)
  x_num = torch_randn(5, 6)
  x_cat = torch_cat(list(
    torch_randint(1, 3, c(5, 1), dtype = torch_long()),
    torch_randint(1, 4, c(5, 1), dtype = torch_long())
  ), dim = 2)

  for (shared in c(TRUE, FALSE)) {
    net = nn_tab_transformer(cardinalities = cardinalities, n_features_num = 6, d_out = 3,
      d_token = 8, depth = 2, n_heads = 2, dim_head = 4, use_shared_categ_embed = shared)
    expect_class(net, "nn_tab_transformer")
    expect_equal(net(x_num = x_num, x_cat = x_cat)$shape, c(5, 3))
  }

  # only categorical features
  net = nn_tab_transformer(cardinalities = cardinalities, n_features_num = 0, d_out = 2,
    d_token = 8, depth = 1, n_heads = 2, dim_head = 4)
  expect_equal(net(x_cat = x_cat)$shape, c(5, 2))

  # only numeric features -> degenerates to layer norm + MLP, no transformer parameters
  net = nn_tab_transformer(cardinalities = integer(0), n_features_num = 6, d_out = 1,
    d_token = 8, depth = 2, n_heads = 2, dim_head = 4)
  expect_equal(net(x_num = x_num)$shape, c(5, 1))
  expect_true(!any(grepl("transformer", names(net$parameters), fixed = TRUE)))
})

test_that("nn_tab_transformer can be built from a task", {
  task = tsk("german_credit")
  net = nn_tab_transformer(task, d_token = 8, depth = 1, n_heads = 2, dim_head = 4)
  # cardinalities are derived in the order in which ingress_categ() selects the features
  features = ingress_categ()$features(task)
  expect_equal(unname(as.integer(net$category_offsets)),
    unname(cumsum(c(0L, head(lengths(task$levels(features))[features], -1L)))))
  expect_equal(net$n_features_num, n_num_features(task))
  expect_equal(net$n_features_categ, n_categ_features(task))
})

test_that("logical features are offset correctly", {
  data = data.frame(
    y = factor(rep(c("a", "b"), 10)),
    lgl = rep(c(TRUE, FALSE), 10),
    fct = factor(rep(c("x", "y", "z", "x"), 5)),
    num = rnorm(20)
  )
  task = as_task_classif(data, target = "y")
  net = nn_tab_transformer(task, d_token = 8, depth = 1, n_heads = 2, dim_head = 4)
  # features in ingress order: fct (3 levels), lgl (0/1 codes -> shifted by one extra)
  expect_equal(unname(as.integer(net$category_offsets)), c(0L, 4L))

  learner = lrn("classif.tab_transformer", epochs = 1, batch_size = 16, d_token = 8, depth = 1,
    n_heads = 2, dim_head = 4)
  expect_error(learner$train(task), regexp = NA)
})

test_that("LearnerTorchTabTransformer works", {
  learner = lrn("classif.tab_transformer", epochs = 1, batch_size = 16, d_token = 8, depth = 1,
    n_heads = 2, dim_head = 4, predict_type = "prob")
  task = tsk("iris")
  learner$train(task)
  pred = learner$predict(task)
  expect_prediction(pred)
  expect_true(!is.null(pred$prob))
  expect_learner_torch(learner, task = tsk("iris"))
})

test_that("task types", {
  learner = lrn("classif.tab_transformer", epochs = 1, batch_size = 16, d_token = 8, depth = 1,
    n_heads = 2, dim_head = 4, mlp_hidden_mults = c(2, 1))
  # multiclass, numeric only
  expect_learner_torch(learner, tsk("iris"))
  # binary, numeric only (degenerate: no categorical features, so no transformer)
  expect_learner_torch(learner, tsk("sonar"))
  # mixed numeric + categorical
  expect_learner_torch(learner, tsk("german_credit"))

  learner_regr = lrn("regr.tab_transformer", epochs = 1, batch_size = 16, d_token = 8, depth = 1,
    n_heads = 2, dim_head = 4, mlp_hidden_mults = c(2, 1))
  expect_learner_torch(learner_regr, tsk("mtcars"))
})

test_that("numeric-only tasks are supported as a degenerate path", {
  learner = lrn("classif.tab_transformer", epochs = 1, batch_size = 16, d_token = 8, depth = 3,
    n_heads = 2, dim_head = 4)
  learner$train(tsk("sonar"))
  expect_prediction(learner$predict(tsk("sonar")))
  # the transformer is not instantiated at all
  expect_true(!any(grepl("transformer", names(learner$network$parameters), fixed = TRUE)))
  expect_equal(learner$network$n_features_categ, 0L)
})

test_that("categorical-only tasks are supported", {
  task = tsk("german_credit")$filter(1:100)
  task$select(ingress_categ()$features(task))
  learner = lrn("classif.tab_transformer", epochs = 1, batch_size = 16, d_token = 8, depth = 1,
    n_heads = 2, dim_head = 4, mlp_hidden_mults = c(1))
  learner$train(task)
  expect_prediction(learner$predict(task))
  # the single input tensor must be routed to `x_cat`, not to `x_num`
  expect_class(learner$network, "nn_tab_transformer_categ_only")
})

test_that("the network is traceable", {
  # 70 rows with batch_size 16 means the last batch has a different size than the traced example
  task = tsk("german_credit")$filter(1:70)
  make = function(jit_trace) {
    lrn("classif.tab_transformer", epochs = 1, batch_size = 16, d_token = 8, depth = 1,
      n_heads = 2, dim_head = 4, mlp_hidden_mults = c(1), jit_trace = jit_trace, shuffle = FALSE,
      seed = 1)
  }
  l_jit = make(TRUE)
  l_jit$train(task)
  l = make(FALSE)
  l$train(task)
  expect_equal(l_jit$predict(task)$response, l$predict(task)$response)
})

test_that("cloning also keeps parameter values", {
  learner = lrn("classif.tab_transformer", depth = 2)
  learnerc = learner$clone(deep = TRUE)
  expect_deep_clone_mlr3torch(learner, learnerc)
  expect_equal(learner$param_set$values$depth, 2)
  expect_equal(learnerc$param_set$values$depth, 2)
})

test_that("error messages", {
  # d_token must leave room for the shared column embedding
  learner = lrn("classif.tab_transformer", epochs = 1, batch_size = 16, d_token = 4, depth = 1,
    n_heads = 2, dim_head = 4)
  expect_error(learner$train(tsk("german_credit")), "must be at least")

  # lazy tensors are not supported
  learner = lrn("classif.tab_transformer", epochs = 1, batch_size = 16, d_token = 8, depth = 1,
    n_heads = 2, dim_head = 4)
  expect_error(learner$train(tsk("lazy_iris")), "lazy_tensor")

  # the module errors informatively when a required input is missing
  net = nn_tab_transformer(cardinalities = c(3), n_features_num = 2, d_out = 1, d_token = 8,
    depth = 1, n_heads = 2, dim_head = 4)
  expect_error(net(x_num = torch_randn(2, 2)), "'x_cat' is required")
  expect_error(net(x_cat = torch_ones(2, 1, dtype = torch_long())), "'x_num' is required")

  # cannot provide both a task and explicit dimensions
  expect_error(nn_tab_transformer(tsk("iris"), n_features_num = 4), "must not be provided")

  # no features at all
  expect_error(nn_tab_transformer(cardinalities = integer(0), n_features_num = 0, d_out = 1),
    "at least one feature")
})
