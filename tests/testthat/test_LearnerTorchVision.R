# different number of classes than the predefined ones
task = as_task_classif(data.table(
  y = as.factor(rep(c("a", "b", "c"), each = 2)),
  x = as_lazy_tensor(torch_randn(6, 3, 64, 64))
), id = "test_task", target = "y")

# Not every network runs on the 64x64 images of `task`, see the `min_size` and `exact_size` columns
# of `torchvision_models`. We use the smallest size that the network accepts, but at least 64.
# The argument must not be called `id`, because data.table would resolve it to the column of that
# name instead of to the argument.
task_for = function(vision_id) {
  info = torchvision_models[list(vision_id), on = "id"]
  size = if (!is.na(info$exact_size)) {
    info$exact_size
  } else {
    max(info$min_size, 64L, na.rm = TRUE)
  }
  if (size == 64L) {
    return(task)
  }
  as_task_classif(data.table(
    y = as.factor(rep(c("a", "b", "c"), each = 2)),
    x = as_lazy_tensor(torch_randn(6, 3, size, size))
  ), id = "test_task", target = "y")
}

test_that("LearnerTorchVision basic checks", {
  alexnet = lrn("classif.alexnet", epochs = 1L, batch_size = 1L, pretrained = FALSE)
  expect_deep_clone_mlr3torch(alexnet, alexnet$clone(deep = TRUE))

  alexnet$train(task)
  expect_class(alexnet$predict(task), "PredictionClassif")

  expect_learner_torch(alexnet, task = task)
  alexnet$id = "a"
  vgg13 = lrn("classif.vgg13", pretrained = FALSE)
  vgg13$id = "a"
  expect_false(alexnet$phash == vgg13$phash)
  expect_true("torchvision" %in% alexnet$packages)

  alexnet = lrn("classif.alexnet", optimizer = "sgd", loss = "cross_entropy",
    callbacks = t_clbk("checkpoint"), epochs = 0, batch_size = 1
  )
  expect_learner(alexnet)
  expect_true("cb.checkpoint.freq" %in% alexnet$param_set$ids())
})

test_that("all torchvision networks are registered and attributed", {
  info = torchvision_models
  expect_names(names(info), permutation.of = c("id", "generator", "label", "arch", "bib", "file",
    "min_size", "exact_size", "jittable"))
  expect_character(info$id, unique = TRUE, any.missing = FALSE)
  expect_character(info$label, unique = TRUE, any.missing = FALSE)
  # a network has either a minimum input size, or exactly one accepted input size, or neither
  expect_integer(info$min_size, lower = 1L)
  expect_integer(info$exact_size, lower = 1L)
  expect_true(all(is.na(info$min_size) | is.na(info$exact_size)))
  expect_true(all(paste0("classif.", info$id) %in% mlr_learners$keys()))

  # every architecture is attributed to a paper for which we have a bibentry
  keys = unique(unlist(lapply(info$bib, torchvision_bib_keys)))
  expect_subset(keys, names(bibentries))

  # every network links to an implementation file of torchvision
  expect_character(info$file, pattern = "^models-.*\\.R$", any.missing = FALSE)

  # the learners get the labels and generators from the table
  learner = lrn("classif.resnet18")
  expect_equal(learner$label, "ResNet-18")
  expect_true(identical(get_private(learner)$.module_generator, torchvision::model_resnet18))
})

test_that("the documentation lists every learner", {
  section = torchvision_learner_section()
  for (id in torchvision_models$id) {
    expect_true(any(grepl(sprintf("`classif.%s`", id), section, fixed = TRUE)))
  }
  # the references cover all cited papers
  refs = torchvision_references()
  expect_true(length(refs) == length(unique(unlist(lapply(torchvision_models$bib, torchvision_bib_keys)))) + 1L)
})

test_that("alexnet", {
  learner = lrn("classif.alexnet", epochs = 0L, batch_size = 2L, pretrained = FALSE)
  learner$train(task, sample(task$nrow, 1L))
  pred = learner$predict(task, sample(task$nrow, 1L))
  expect_class(pred, "PredictionClassif")
})

test_that("maxvit gets a class for replace_head()", {
  # the module returned by torchvision has no class of its own
  network = maxvit_generator(pretrained = FALSE, num_classes = 10L)
  expect_class(network, "maxvit")
  expect_equal(replace_head(network, 3L)$classifier$`5`$out_features, 3L)
})

test_that("aux_logits and aux_weight are only exposed by inception_v3", {
  learner = lrn("classif.inception_v3")
  expect_true(all(c("aux_logits", "aux_weight") %in% learner$param_set$ids()))
  expect_false(any(c("aux_logits", "aux_weight") %in% lrn("classif.resnet18")$param_set$ids()))
  # the auxiliary classifier is off by default
  expect_null(learner$param_set$values$aux_logits)
})

test_that("inception_v3 has no auxiliary classifier by default", {
  # otherwise the network returns a list of (logits, aux_logits) during training, which the
  # default loss cannot handle
  network = inception_v3_generator(pretrained = FALSE, num_classes = 3L)
  expect_false(network$aux_logits)
  # the parameters of the auxiliary classifier are deregistered, not just ignored. Assigning
  # NULL would leave them registered and hand them to the optimizer without a gradient.
  reference = inception_v3_generator(pretrained = FALSE, aux_logits = FALSE, num_classes = 3L)
  expect_equal(length(network$parameters), length(reference$parameters))

  network$train()
  expect_class(network(torch_randn(1, 3, 75, 75)), "torch_tensor")
})

test_that("nn_aux_loss combines the predictions of the auxiliary classifiers", {
  base = nn_cross_entropy_loss()
  loss = nn_aux_loss(base, aux_weight = 0.4)
  target = torch_tensor(c(1L, 2L), dtype = torch_long())
  main = torch_randn(2, 3)
  aux = torch_randn(2, 3)

  # the loss is only applied during training, where the auxiliary classifiers are active, so it
  # only has to handle a list. A tensor would otherwise be indexed by row without an error.
  expect_error(loss(main, target), "list")

  expect_equal(as.numeric(loss(list(main, aux), target)),
    as.numeric(base(main, target) + 0.4 * base(aux, target)), tolerance = 1e-6)

  # a weight of zero reduces to the primary prediction
  expect_equal(as.numeric(nn_aux_loss(base, aux_weight = 0)(list(main, aux), target)),
    as.numeric(base(main, target)), tolerance = 1e-6)

  expect_error(nn_aux_loss(base, aux_weight = -1), "aux_weight")
})

test_that("inception_v3 wraps the configured loss when aux_logits is enabled", {
  task_aux = as_task_classif(data.table(
    y = as.factor(rep(c("a", "b", "c"), each = 2)),
    x = as_lazy_tensor(torch_randn(6, 3, 299, 299))
  ), id = "test_task_aux", target = "y")

  # the user configures an ordinary loss, the learner wraps it
  learner = lrn("classif.inception_v3", pretrained = FALSE, aux_logits = TRUE, epochs = 1L,
    batch_size = 2L, loss = t_loss("cross_entropy"), aux_weight = 0.2, predict_type = "prob")
  loss_fn = get_private(learner)$.loss_fn(task_aux, learner$param_set$values)
  expect_class(loss_fn, "nn_aux_loss")
  expect_class(loss_fn$base_loss, "nn_cross_entropy_loss")
  expect_equal(loss_fn$aux_weight, 0.2)

  # without the auxiliary classifier the loss is left alone
  learner_plain = lrn("classif.inception_v3", pretrained = FALSE, loss = t_loss("cross_entropy"))
  expect_class(get_private(learner_plain)$.loss_fn(task_aux, learner_plain$param_set$values),
    "nn_cross_entropy_loss")
})

test_that("jittable is forwarded, so jit_trace is available exactly for traceable networks", {
  # `LearnerTorchVision` has to pass `jittable` on to `super$initialize()`, otherwise the whole
  # `jittable` column of `torchvision_models` has no effect and no network can be traced.
  for (i in seq_len(nrow(torchvision_models))) {
    vision_id = torchvision_models$id[i]
    expect_equal("jit_trace" %in% lrn(paste0("classif.", vision_id))$param_set$ids(),
      torchvision_models$jittable[i], info = vision_id)
  }
})

test_that("inception_v3 is not traceable", {
  # with an enabled auxiliary classifier the network returns a list of predictions, which
  # torch::jit_trace() cannot represent, so the learner offers no jit_trace parameter at all
  expect_false("jit_trace" %in% lrn("classif.inception_v3")$param_set$ids())
  expect_false(torchvision_models[list("inception_v3"), on = "id"]$jittable)
})

# these tests are run the CI, but they should basically never fail, so
# we skip them in the local run
# models are also cached in the CI, so it is not too slow
skip_if(!identical(Sys.getenv("INCLUDE_IGNORED"),  "1"), "Slow vision tests")

# ViT-L/16 and ViT-H/14 need more memory than the smallest CI runners have (~7 GB): they have
# 300M resp. 630M parameters and, because of their small patch size, produce long token sequences
# (197 resp. 1370 tokens). The remaining vision transformers cover the same code path, so we skip
# these two instead of letting them run out of memory.
memory_heavy = c("vit_l_16", "vit_h_14")

# Train every other network from scratch. This needs no downloads and therefore covers all of them.
for (vision_id in torchvision_models$id) {
  test_that(paste0("network can be trained: ", vision_id), {
    skip_if(vision_id %in% memory_heavy, "Network does not fit into the memory of small runners")
    learner = lrn(paste0("classif.", vision_id), epochs = 1L, batch_size = 2L, pretrained = FALSE)
    t = task_for(vision_id)
    learner$train(t, sample(t$nrow, 2L))
    pred = learner$predict(t, sample(t$nrow, 1L))
    expect_class(pred, "PredictionClassif")
  })
}

# Loading the pretrained weights requires a download, so we only cover one network per
# architecture here. This is what exercises the replace_head() methods.
pretrained_ids = c("alexnet", "convnext_tiny_1k", "efficientnet_b0", "efficientnet_v2_s",
  "inception_v3", "maxvit", "mobilenet_v2", "mobilenet_v3_small", "resnet18", "resnext50_32x4d",
  "vgg11", "vit_b_16", "wide_resnet50_2")

test_that("one network per architecture is covered by the pretrained tests", {
  expect_set_equal(unique(torchvision_models[list(pretrained_ids), on = "id"]$arch),
    unique(torchvision_models$arch))
})

for (vision_id in pretrained_ids) {
  test_that(paste0("pretrained network can be fine-tuned: ", vision_id), {
    learner = lrn(paste0("classif.", vision_id), epochs = 1L, batch_size = 2L, pretrained = TRUE,
      predict_type = "prob")
    t = task_for(vision_id)
    # two rows and one epoch, so that a single fine-tuning step is actually performed
    learner$train(t, sample(t$nrow, 2L))
    pred = learner$predict(t, sample(t$nrow, 1L))
    expect_class(pred, "PredictionClassif")
    # replace_head() gave the network a head that matches the number of classes of the task
    expect_equal(ncol(pred$prob), length(t$class_names))
  })
}

test_that("inception_v3 can be trained with an auxiliary classifier", {
  # the auxiliary classifier pools and convolves the feature map further than the main
  # classifier, which raises the minimum input size from 75x75 to 299x299
  task_aux = as_task_classif(data.table(
    y = as.factor(rep(c("a", "b", "c"), each = 2)),
    x = as_lazy_tensor(torch_randn(6, 3, 299, 299))
  ), id = "test_task_aux", target = "y")

  learner = lrn("classif.inception_v3", pretrained = FALSE, aux_logits = TRUE, epochs = 1L,
    batch_size = 2L, predict_type = "prob")
  learner$train(task_aux)
  expect_true(learner$network$aux_logits)

  # both heads predict the number of classes of the task, not the 1000 of the generator default
  learner$network$train()
  out = learner$network(torch_randn(1, 3, 299, 299))
  expect_list(out, len = 2L)
  expect_equal(dim(out$logits)[2L], 3L)
  expect_equal(dim(out$aux_logits)[2L], 3L)

  # only the primary prediction is returned when predicting
  pred = learner$predict(task_aux)
  expect_class(pred, "PredictionClassif")
  expect_equal(ncol(pred$prob), 3L)

  # measures_train scores the primary prediction, see learner_torch_train(). Without this the
  # training loop calls $detach() on a list and fails.
  learner_measured = lrn("classif.inception_v3", pretrained = FALSE, aux_logits = TRUE,
    epochs = 1L, batch_size = 2L, measures_train = msrs("classif.acc"))
  expect_no_error(learner_measured$train(task_aux))
})

test_that("ctx exposes the primary prediction and the complete network output", {
  task_aux = as_task_classif(data.table(
    y = as.factor(rep(c("a", "b", "c"), each = 2)),
    x = as_lazy_tensor(torch_randn(6, 3, 299, 299))
  ), id = "test_task_aux", target = "y")

  seen = new.env(parent = emptyenv())
  cb = torch_callback("test_yhats",
    on_batch_end = function() {
      seen$y_hat = self$ctx$y_hat
      seen$y_hats = self$ctx$y_hats
    }
  )

  # with an auxiliary classifier `y_hats` is the list, `y_hat` its first element
  learner = lrn("classif.inception_v3", pretrained = FALSE, aux_logits = TRUE, epochs = 1L,
    batch_size = 2L, callbacks = cb)
  learner$train(task_aux)
  expect_list(seen$y_hats, len = 2L)
  expect_class(seen$y_hat, "torch_tensor")
  expect_true(torch_equal(seen$y_hat, seen$y_hats[[1L]]))

  # without one, both are the same tensor
  learner_plain = lrn("classif.inception_v3", pretrained = FALSE, epochs = 1L, batch_size = 2L,
    callbacks = cb)
  learner_plain$train(task_aux)
  expect_class(seen$y_hats, "torch_tensor")
  expect_true(torch_equal(seen$y_hat, seen$y_hats))
})
