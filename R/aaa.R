register_po = function(name, constructor, metainf = NULL) {
  if (name %in% names(mlr3torch_pipeops)) stopf("pipeop %s registered twice", name)
  mlr3torch_pipeops[[name]] = list(constructor = constructor, metainf = substitute(metainf))
}

register_learner = function(.name, .constructor, ...) {
  assert_multi_class(.constructor, c("function", "R6ClassGenerator"))
  if (is.function(.constructor)) {
    mlr3torch_learners[[.name]] = list(fn = .constructor, prototype_args = list(...))
    return(NULL)
  }
  # learners are registered as "<task_type>.<key>", e.g. "classif.mlp" or "torch_supervised.module"
  task_type = assert_choice(sub("\\..*$", "", .name), c("classif", "regr", names(mlr3torch_task_types)))
  # What I am doing here:
  # The problem is that we wan't to set the task_type when creating the learner from the dictionary
  # The initial idea was to add functions function(...) LearnerClass$new(..., task_type = "<task-type>")
  # This did not work because mlr3misc does not work with ... arguments (... arguments are not
  # passed further to the initialize method)
  # For this reason, we need this hacky solution here, might change in the future in mlr3misc
  fn = crate(function() {
    invoke(.constructor$new, task_type = task_type, .args = as.list(match.call()[-1]))
  }, .constructor, task_type, .parent = topenv())
  fmls = formals(.constructor$public_methods$initialize)
  fmls$task_type = NULL
  formals(fn) = fmls
  if (.name %in% names(mlr3torch_learners)) stopf("learner %s registered twice", .name)
  mlr3torch_learners[[.name]] = list(fn = fn, prototype_args = list(...))
}

register_task = function(name, constructor) {
  if (name %in% names(mlr3torch_tasks)) stopf("task %s registered twice", name)
  mlr3torch_tasks[[name]] = constructor
}

# The general-purpose task types of mlr3torch, mapped to the `Task` class that implements them.
# A learning problem that is neither classification nor regression is an *instance* of one of these
# two rather than a task type of its own, see `TaskTorch` and `TaskTorchUnsupervised`.
mlr3torch_task_types = c(torch_supervised = "TaskTorch", torch_unsupervised = "TaskTorchUnsupervised")

mlr3torch_pipeops = new.env()
mlr3torch_learners = new.env()
mlr3torch_tasks = new.env()
