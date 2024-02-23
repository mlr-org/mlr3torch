#' @param dataset_shapes (named `list()` of (`integer()` or `NULL`))\cr
#'   The shapes of the output.
#'   Names are the elements of the list returned by the dataset.
#'   If the shape is not `NULL` (unknown, e.g. for images of different sizes) the first dimension must be `NA` to
#'   indicate the batch dimension.
