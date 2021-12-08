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
#' @return [`data.table`] with columns "image" and "class" (the class).
#' Column "image" has additional class "imageuri".
#' @export
#' @examples
#' \dontrun{
#' df_from_imagenet_dir(c(
#'   "/opt/example-data/imagenette2-160/train/",
#'   "/opt/example-data/imagenette2-160/val/"
#' ))
#' }
df_from_imagenet_dir <- function(dirs) {
  img_file <- fs::dir_ls(dirs, recurse = TRUE, type = "file")
  img_class <- fs::path_file(fs::path_dir(img_file))
  class(img_file) <- c("imageuri", "character")

  data.table::data.table(
    class = factor(img_class),
    image = img_file
  )
}
