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
#' @return [`data.frame`]\cr
#'   with columns "image" and "class" (the class).
#' @export
#' @author Florian Pfisterer
#' @examples
#' \dontrun{
#' df_from_imagenet_dir(c("/opt/example-data/imagenette2-160/train/", "/opt/example-data/imagenette2-160/val/"))
#'
#' }
df_from_imagenet_dir = function(dirs) {
  rbindlist(map(dirs, function(y) {
    dt = rbindlist(map(list.dirs(y, recursive = FALSE), function(x) {
      data.table(
        class = basename(x),
        image = list.files(x, full.names = TRUE)
      )
    }))
    set(dt, j = "class", value = as.factor(dt$class))
    return(dt)
  }))
}


