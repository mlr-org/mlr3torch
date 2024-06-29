#' <% task_types_vec = strsplit(task_types, ", ")[[1]] %>
#' <% param_vals = if (exists("param_vals", inherits = FALSE)) param_vals else character(0) %>
#' <% id = paste0(task_types_vec[1], ".", name)%>
#' <% task_id = default_task_id(lrn(id)) %>
#'
#' @examplesIf torch::torch_is_installed()
#' @examples
#' # Define the Learner and set parameter values
#' <%= sprintf("learner = lrn(\"%s\")", id)%>
#' learner$param_set$set_values(
#'   epochs = 1, batch_size = 16, device = "cpu"<%= if (length(param_vals)) "," else character()%>
#'   <%= if (length(param_vals)) paste0(param_vals, collapse = ",\n ") else character()%>
#' )
#'
#' # Define a Task
#' <%= sprintf("task = tsk(\"%s\")", task_id)%>
#'
#' # Create train and test set
#' <%= sprintf("ids = partition(task)")%>
#'
#' # Train the learner on the training ids
#' <%= sprintf("learner$train(task, row_ids = ids$train)")%>
#'
#' # Make predictions for the test rows
#' <%= sprintf("predictions = learner$predict(task, row_ids = ids$test)")%>
#'
#' # Score the predictions
#' predictions$score()
