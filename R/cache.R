#' CACHE
#' This keeps track of the cache versions.
#' When incremented in a release, it ensures that the previous cache gets flushed, thereby
#' allowing to easily change the caching mechanism / structure of files in the future.
#' @noRd
CACHE = new.env(hash = FALSE, parent = emptyenv())

CACHE$versions = list(
  datasets = 1L
)

CACHE$initialized = character()

#' Returns the cache directory
#' @param cache Whether to cache.
#' @noRd
get_cache_dir = function(cache = NULL) {
  if (is.null(cache)) cache = getOption("mlr3torch.cache", FALSE)
  assert_true(is.logical(cache) || is.character(cache))
  if (isFALSE(cache)) {
    return(FALSE)
  }
  if (!is.character(cache)) {
    cache = R_user_dir("mlr3torch", "cache")
  }

  assert(check_directory_exists(cache), check_path_for_output(cache))
  normalizePath(cache, mustWork = FALSE)
}

#' Initializes the cache directory.
#' When a cached is initialized in a session, it is added to the `CACHE$initialized` list and we trust
#' it without checking the cache versions each time.
#' Otherwise we compare the written cache versions for the subfolders like `datasets` with the
#' current CACHE versions of the mlr3torch package. If they differ, we flush the cache and initialize
#' a new folder with the updated cache version.
#'
#' @noRd
initialize_cache = function(cache_dir) {
  if (isFALSE(cache_dir) || (file.exists(cache_dir) && normalizePath(cache_dir, mustWork = FALSE) %in%
      CACHE$initialized)) {
    lg$debug("Skipping initialization of cache", cache_dir = cache_dir)
    return(TRUE)
  }

  require_namespaces("jsonlite", "The following packages are required for caching: %s")
  cache_file = file.path(cache_dir, "version.json")
  write_cache_file = FALSE

  if (dir.exists(cache_dir)) {
    if (file.exists(cache_file)) {
      cache_versions = jsonlite::fromJSON(cache_file)
      for (type in intersect(names(cache_versions), names(CACHE$versions))) {
        if (cache_versions[[type]] != CACHE$versions[[type]]) {
          lg$debug("Invalidating cache dir due to a version mismatch", path = file.path(cache_dir, type))

          unlink(file.path(cache_dir, type), recursive = TRUE)
          write_cache_file = TRUE
        }
      }
    } else {
      stopf("Cache directory '%s' was not initialized by mlr3torch", cache_dir)
    }
  } else {
    dir.create(cache_dir, recursive = TRUE)
    write_cache_file = TRUE
  }

  if (write_cache_file) {
    lg$debug("Writing cache version information", path = cache_file)
    writeLines(jsonlite::toJSON(CACHE$versions, auto_unbox = TRUE), con = cache_file)
  }

  CACHE$initialized = c(CACHE$initialized, normalizePath(cache_dir, mustWork = FALSE))

  return(TRUE)
}

cached = function(constructor, type, name) {
  cache_dir = get_cache_dir()
  initialize_cache(cache_dir)
  assert_choice(type, names(CACHE$versions))

  do_caching = !isFALSE(cache_dir)

  # Even when we don't cache, we need to store the data somewhere
  path = normalizePath(if (do_caching) file.path(cache_dir, type, name) else tempfile(), mustWork = FALSE)

  if (do_caching && dir.exists(path)) {
    # we cache and there is a cache hit
    data = try(readRDS(file.path(path, "data.rds")), silent = TRUE)
    if (!inherits(data, "try-error")) {
      return(list(data = data, path = path))
    }
    lg$debug("Cache hit failed, removing cache", path = path)
    lapply(list.files(path, full.names = TRUE), unlink)
  }
  # We either don't cache, there is no cache hit or cache retrieval failed
  data = try({
    path_raw = file.path(path, "raw")
    if (!dir.exists(path_raw)) {
      dir.create(path_raw, recursive = TRUE)
    }
    constructor(path_raw)
    }, silent = TRUE
  )
  if (inherits(data, "try-error")) {
    # in case anything goes wrong during the construction we need to clean up.
    # Otherwise we might get cache hits on corrupt folders
    unlink(path, recursive = TRUE)
    stop(data)
  }

  # now path/raw exists

  if (do_caching) {
    # store the processed data in case there is a cache hit, so next time we don't need the postprocessing
    # that comes after downloading the data
    saveRDS(data, file = file.path(path, "data.rds"))
  }
  list(data = data, path = path)
}

clear_mlr3torch_cache = function() {
  if (isFALSE(get_cache_dir())) {
    catn("No cache directory set.")
    return(FALSE)
  }
  unlink(get_cache_dir(), recursive = TRUE)
  CACHE$initialized = setdiff(CACHE$initialized, normalizePath(get_cache_dir(), mustWork = FALSE))
  return(TRUE)
}
