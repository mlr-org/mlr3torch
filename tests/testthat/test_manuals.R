test_that("All PipeOpTorch have a consistent help page", {
  tmp = list.files("man")
  tmp = tmp[grepl("^mlr_pipeops", tmp)]
  po_manuals = lapply(tmp, function(x) readLines(file.path("man", x)))
  names(po_manuals) = tmp

  xs = c(
    "Construction",
    "Input and Output Channels",
    "State",
    "Parameters",
    "Fields",
    "Methods",
    "Internals"
  )

  # Note that we can't test for the order of the entries, because of https://github.com/r-lib/roxygen2/issues/900

  res = imap(po_manuals,
    function(po_manual, nm) {
      matches = map_lgl(xs,
        function(x) {
          x = grep(sprintf("\\\\section\\{%s\\}\\{", x), po_manual)
          length(x) == 1
        }
      )
      names(matches) = xs
      return(matches)
    }
  )
  incomplete = Filter(function(x) !all(x), res)
  expect_true(length(incomplete) == 0)
})

test_that("All Learners have a consistent help page", {
  expect_true(FALSE)
})
