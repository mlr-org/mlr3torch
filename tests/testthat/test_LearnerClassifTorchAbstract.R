test_that("assemble and dissamble work.", {
  task = tsk("iris")
  f = function() {
    devtools::load_all("~/mlr/mlr3torch")
    learner = mlr3::lrn("classif.mlp",
      layers = 2L,
      p = 0.2,
      batch_size = 16L,
      epochs = 1L,
      d_hidden = 10,
      activation = "relu",
      optimizer = "adam"
    )
    task = mlr3::tsk("iris")

    learner$train(task)
    learner$serialize()
    saveRDS(learner, "~/.lol/loeerna.R")

  }
  callr::r(f)
  learner = readRDS("~/.lol/loeerna.R")
  learner$unserialize()
  learner$train(task)

  dir = tempfile(fileext = ".rds")
  saveRDS(learner, dir)
  learner_ = readRDS(dir)

  saveRDS(learner$s)
}
