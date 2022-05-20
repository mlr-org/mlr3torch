#' Create a dataframe from a directory with the imagenet
#' directory structure.
#'
#' The imagenet directory structure is laid out as follows
#' Top-level dirs: `train`, `test`, `valid` \cr
#' (you should provide path to those dirs as input) \cr
#' Mid-level dirs: `class1`, `class2`, `...` \cr
#' One directory for each class. The folders directly contain the images.
#' @param dirs [`character`]\cr
#'   List of dirs to create dataframes from.
#' @return [`data.table`] with columns "uri" and "target" (the class).
#' Column "uri" has additional class "imageuri".
#' @export
#' @examples
#' \dontrun{
#' df_from_imagenet_dir(c(
#'   "/opt/example-data/imagenette2-160/train/",
#'   "/opt/example-data/imagenette2-160/val/"
#' ))
#'
#' # Only supplying the parent folder also works
#' df_from_imagenet_dir(
#'   "/opt/example-data/imagenette2-160/"
#' )
#' }
df_from_imagenet_dir = function(dirs) {

  # known / valid image extensions to avoid e.g. spurious csv files
  img_ext = c("jpg", "jpeg", "png")

  img_file = fs::dir_ls(dirs, recurse = TRUE, type = "file")

  # Subset to valid image files (by extension at least)
  img_file = img_file[which(tolower(fs::path_ext(img_file)) %in% img_ext)]

  img_class = fs::path_file(fs::path_dir(img_file))
  class(img_file) = c("imageuri", "character")

  data.table::data.table(
    target = factor(img_class),
    uri = img_file
  )
}


#' Create a torch dataset from a data.table of images
#'
#' @param df Table of image URIs and targets to create dataset from, as
#'   returned by e.g. `df_from_imagenet_dir()`. Assumed to have columns
#'   `'target'` and `'uri'`.
#' @param row_ids Integer vector specifying which rows of `df` to use.
#' @param transform Function of transformation to apply to input images.
#'    By default applies `torchvision::transform_to_tensor()` only, yet
#'    model-specific transformations (e.g. resizing) or data augmentation
#'    may be necessary.
#' @param target_transform Function of optional transformations applied to the
#'   `'target'`, which will be coerced via `as.integer()` and
#'   `torch::torch_tensor()` before any `target_transform()` is applied.
#' @export
#' @examples
#' \dontrun{
#' # Dataset of all imagenette160 images
#' image_dt = df_from_imagenet_dir("/opt/example-data/imagenette2-160/")
#' img_ds = img_dataset(image_dt)
#'
#' # Datasets for train- and validation images
#' # Inferring set from file path (URI)
#' train_ids = grepl("/train/", image_dt$uri)
#' val_ids = grepl("/val/", image_dt$uri)
#'
#' img_train_ds = img_dataset(image_dt, row_ids = train_ids)
#' img_val_ds = img_dataset(image_dt, row_ids = val_ids)
#'
#' # Adding transformations, including device placement
#' device = if (torch::cuda_is_available()) "cuda" else "cpu"
#'
#' to_device = function(x, device) {
#'   x$to(device = device)
#' }
#'
#' img_transform = function(x) {
#'   x %>%
#'     torchvision::transform_to_tensor() %>%
#'     to_device(device) %>%
#'     torchvision::transform_resize(c(64, 64))
#' }
#'
#' img_train_ds = img_dataset(image_dt, row_ids = train_ids, img_transform)
#' img_val_ds = img_dataset(image_dt, row_ids = val_ids, img_transform)
#' }
img_dataset = torch::dataset(
  "image-dataset",

  # Input is the result of df_from_imagenet_dir()
  # classes are converted to integers since they are stored as factors
  # which should make class id <-> name conversion easy(ish)
  initialize = function(df, row_ids = NULL,
    transform = torchvision::transform_to_tensor,
    target_transform = NULL) {

    if (!is.null(row_ids)) df <- df[row_ids, ]

    self$num_classes = length(unique(df[["target"]]))

    self$uri = df[["uri"]]
    self$target = torch::torch_tensor(as.integer(df[["target"]]))
    self$transform = transform
    self$target_transform = target_transform
  },

  # Get individual item based on index only
  .getitem = function(index) {

    # target <- torch_tensor(self$class[index])
    target = as.integer(self$target[index])
    img = torchvision::magick_loader(self$uri[[index]])

    if (!is.null(self$transform)) img <- self$transform(img)
    if (!is.null(self$target_transform)) target <- self$target_transform(target)

    list(x = img, y = target)
  },

  # It's optional, but helpful to define the .length method returning
  # the number of elements in the dataset. This is needed if you want
  # to shuffle your dataset.
  .length = function() {
    length(self$target)
  }
)
