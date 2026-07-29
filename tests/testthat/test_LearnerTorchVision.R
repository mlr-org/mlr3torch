# different number of classes than the predefined ones
task = as_task_classif(data.table(
  y = as.factor(rep(c("a", "b", "c"), each = 2)),
  x = as_lazy_tensor(torch_randn(6, 3, 64, 64))
), id = "test_task", target = "y")

# some networks (the vision transformers, MaxViT, Inception v3) require a larger input,
# see the `input` column of `torchvision_models`
task_for = function(id) {
  input = torchvision_models[list(id), on = "id"]$input
  size = if (is.na(input)) 64L else if (input == "518x518") 518L else if (input == "224x224") 224L else 96L
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
  expect_names(names(info),
    permutation.of = c("id", "generator", "label", "arch", "bib", "file", "input", "jittable"))
  expect_character(info$id, unique = TRUE, any.missing = FALSE)
  expect_character(info$label, unique = TRUE, any.missing = FALSE)
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

test_that("inception_v3 has no auxiliary classifier", {
  # otherwise the network returns a list of (logits, aux_logits) during training, which the
  # training loop cannot handle
  network = inception_v3_generator(pretrained = FALSE, num_classes = 3L)
  expect_false(network$aux_logits)
  network$train()
  expect_class(network(torch_randn(2, 3, 96, 96)), "torch_tensor")
})

test_that("maxvit gets a class for replace_head()", {
  # the module returned by torchvision has no class of its own
  network = maxvit_generator(pretrained = FALSE, num_classes = 10L)
  expect_class(network, "maxvit")
  expect_equal(replace_head(network, 3L)$classifier$`5`$out_features, 3L)
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
    learner = lrn(paste0("classif.", vision_id), epochs = 0L, batch_size = 2L, pretrained = FALSE)
    t = task_for(vision_id)
    learner$train(t, sample(t$nrow, 1L))
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
    learner = lrn(paste0("classif.", vision_id), epochs = 0L, batch_size = 2L, pretrained = TRUE,
      predict_type = "prob")
    t = task_for(vision_id)
    learner$train(t, sample(t$nrow, 1L))
    pred = learner$predict(t, sample(t$nrow, 1L))
    expect_class(pred, "PredictionClassif")
    # replace_head() gave the network a head that matches the number of classes of the task
    expect_equal(ncol(pred$prob), length(t$class_names))
  })
}
