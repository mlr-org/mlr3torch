# Test num_threads
library(mlr3)
library(mlr3torch)
library(mlr3benchmark)

lrn <- lrn("classif.torch.tabnet", epochs = 5L, num_threads = 1L)

# Without parallelization -------------------------------------------------
cli::cli_h1("CV without parallelisation")
tictoc::tic()
set.seed(247)
torch::torch_manual_seed(247)
rr_single <- resample(
  task = tsk("spam"),
  learner = lrn,
  resampling = rsmp("cv", folds = 5)
)
tictoc::toc()
saveRDS(rr_single, here::here("attic", glue::glue("rr_single_{format(Sys.time(), format = '%Y%m%d%H%M%S')}.rds")))

# With multisession -------------------------------------------------------
cli::cli_h1("CV with plan('multisession')")
tictoc::tic()
future::plan("multisession")
set.seed(247)
torch::torch_manual_seed(247)
rr_multisession <- resample(
  task = tsk("spam"),
  learner = lrn,
  resampling = rsmp("cv", folds = 5)
)
tictoc::toc()
saveRDS(rr_single, here::here("attic", glue::glue("rr_multisession_{format(Sys.time(), format = '%Y%m%d%H%M%S')}.rds")))

# With multicore ----------------------------------------------------------

# Error in (function (self, inputs, gradient, retain_graph, create_graph)  :
# Unable to handle autograd's threading in combination with fork-based multiprocessing. See https://github.com/pytorch/pytorch/wiki/Autograd-and-Fork

# future::plan("multicore")
# set.seed(247)
# rr_multicore <- resample(
#   task = tsk("spam"),
#   learner = lrn,
#   resampling = rsmp("cv", folds = 5)
# )

if (interactive()) {

  # TODO: Figure out how to Reduce() an all.equal over sub-elements or something
  cli::cli_h1("Checking sequential resampling")
  rr_singles <- purrr::map(fs::dir_ls(here::here("attic"), glob = "*rr_single*rds"), readRDS)

  cli::cli_text("Checking if predictions are the same")
  all.equal(rr_singles[[1]]$predictions(), rr_singles[[2]]$predictions())

  cli::cli_text("Checking if scored CE is the same")
  all.equal(rr_singles[[1]]$score()[["classif.ce"]], rr_singles[[2]]$score()[["classif.ce"]])

  cli::cli_h1("Checking multisession resampling")
  rr_multisess <- purrr::map(fs::dir_ls(here::here("attic"), glob = "*rr_multisession*rds"), readRDS)

  cli::cli_text("Checking if predictions are the same")
  all.equal(rr_multisess[[1]]$predictions(), rr_multisess[[2]]$predictions())

  cli::cli_text("Checking if scored CE is the same")
  all.equal(rr_multisess[[1]]$score()[["classif.ce"]], rr_multisess[[2]]$score()[["classif.ce"]])

}
