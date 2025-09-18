#!/usr/bin/env Rscript
# find_non_ascii.R
# Recursively scans only .R files under a path and reports those containing any non-ASCII byte (> 0x7F).

args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args) >= 1) args[1] else "."

# Return TRUE iff file contains any non-ASCII byte. Reads in chunks to handle large files.
has_non_ascii <- function(path, chunk_size = 1048576L) {
  tryCatch({
    con <- file(path, open = "rb")
    on.exit(close(con), add = TRUE)
    repeat {
      buf <- readBin(con, what = "raw", n = chunk_size)
      if (length(buf) == 0L) break
      if (any(buf > as.raw(0x7F))) return(TRUE)
    }
    FALSE
  }, error = function(e) {
    message(sprintf("WARN: Could not read '%s' (%s). Skipping.", path, conditionMessage(e)))
    NA
  })
}

# Collect files (excluding directories). Includes hidden files.
all_paths <- list.files(
  path = root,
  recursive = TRUE,
  all.files = TRUE,
  include.dirs = FALSE,
  full.names = TRUE,
  no.. = TRUE
)

# Keep only regular files with extension .R (case-insensitive)
fi <- file.info(all_paths, extra_cols = FALSE)
files <- all_paths[!is.na(fi$isdir) & !fi$isdir]
ext <- tolower(tools::file_ext(files))
files <- files[ext == "r"]

if (length(files) == 0L) {
  cat(sprintf("No .R files found under: %s\n", normalizePath(root, mustWork = FALSE)))
  quit(status = 0)
}

# Scan
res <- vapply(files, has_non_ascii, logical(1), USE.NAMES = TRUE)

# Report
offenders <- names(res)[!is.na(res) & res]
errors    <- names(res)[is.na(res)]

if (length(offenders)) {
  cat("R files containing non-ASCII bytes:\n")
  cat(paste0("  - ", offenders), sep = "\n")
} else {
  cat("No non-ASCII bytes found in .R files.\n")
}

if (length(errors)) {
  cat("\nFiles skipped due to read errors:\n")
  cat(paste0("  - ", errors), sep = "\n")
}

# Exit code: 0 if none found, 1 if any found (useful in CI)
quit(status = if (length(offenders)) 1 else 0)
