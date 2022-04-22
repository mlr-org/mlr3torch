#' Convert a task to torch dataloader(s)
#'
#' @param task An object inheriting from [Task][mlr3::Task]. Construction can be
#' aided by [df_from_imagenet_dir()].
#' @param valid_split `[0.2]` Proportion of rows in task to use as validation set.
#' If `0`, no validation dataloader will be created.
#' @param transform_train A function used to transform images, optionally
#' including data augmentation.
#' Must begin with [`torchvision::transform_to_tensor()`] and manually move
#' to GPU if available.
#' @param transform_val Same as `transform_train` but applied to the validation
#' data, typically only including necessary transformations. Ignored if
#' `valid_split` is `0`.
#' @param batch_size,drop_last Passed to [`torch::dataloader()`].
#'
#' @return A list containing elements `train` and `val`,
#' each containing the respective [`dataloader`][torch::dataloader] objects.
#' If `valid_split == 0`, the list contains a single dataloader `"train"`.
#' @export
#'
#' @examples
#' \dontrun{
#' dls = make_dl_from_task(
#'   img_task,
#'   valid_split = 0.2,
#'   transform_train = img_transforms,
#'   transform_val = img_transforms
#' )
#' # Verify counts in dataloaders
#' dls$train$.length()
#' dls$val$.length()
#'
#' # Should correspond to valid_split
#' dls$val$.length() / (dls$train$.length() + dls$val$.length())
#'
#' # No validation dataloader
#' dls = make_dl_from_task(
#'   img_task,
#'   valid_split = 0,
#'   transform_train = img_transforms
#' )
#' }
make_dl_from_task = function(
  task,
  valid_split = 0,
  batch_size = 32L,
  transform_train,
  transform_val = NULL,
  drop_last = TRUE
) {
  assert_task(task)
  assert_double(valid_split)
  assert_integerish(batch_size)
  assert_function(transform_train)
  # If there's a transform_val it has to be a function
  if (!is.null(transform_val)) checkmate::assert_function(transform_val)
  # transform_val has to be provided only if validation split is used
  if (valid_split > 0 & is.null(transform_val)) {
    stop("No validation transformation function provided")
  }
  # if transform_val set but no validation split is used the user should know?
  if (valid_split == 0 & !is.null(transform_val)) {
    warning("Validation transformation is ignored when valid_split is 0")
  }
  assert_logical(drop_last)

  # Allow for no validation split to convert test sets to dl later
  if (valid_split > 0) {
    train_ids = sample(task$row_ids,
      size = ceiling(task$nrow * (1 - valid_split)))
    valid_ids = setdiff(task$row_ids, train_ids)
  } else {
    train_ids = task$row_ids
  }

  train_ds = img_dataset(task$data(), row_ids = train_ids,
    transform = transform_train)
  train_dl = torch::dataloader(train_ds, batch_size = batch_size,
    shuffle = TRUE, drop_last = drop_last)

  dls = list(train = train_dl)

  if (valid_split > 0) {
    valid_ds = img_dataset(task$data(), row_ids = valid_ids,
      transform = transform_val)
    valid_dl = torch::dataloader(valid_ds, batch_size = batch_size,
      shuffle = FALSE, drop_last = drop_last)

    dls$val = valid_dl
  }

  dls

}
