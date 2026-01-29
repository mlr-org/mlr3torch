# Script to extract R code from paper.tex
# Extracts lines between \begin{CodeInput} and \end{CodeInput}
# Only keeps lines starting with "R>" or "+" and removes these prefixes

# Read the LaTeX file
tex_file <- readLines("paper/paper.tex")

# Initialize variables to track state
in_code_block <- FALSE
code_lines <- character(0)

# Process each line
for (line in tex_file) {
  # Check if we're entering a code block
  if (grepl("\\\\begin\\{CodeInput\\}", line)) {
    in_code_block <- TRUE
    next
  }

  # Check if we're exiting a code block
  if (grepl("\\\\end\\{CodeInput\\}", line)) {
    in_code_block <- FALSE
    # Add an empty line after each code block
    code_lines <- c(code_lines, "")
    next
  }

  # If we're in a code block, check if the line starts with R> or +
  if (in_code_block) {
    if (grepl("^R>\\s+", line)) {
      # Remove the "R>" prefix and one following whitespace if it exists
      clean_line <- sub("^R>\\s", "", line)
      code_lines <- c(code_lines, clean_line)
    } else if (grepl("^\\+\\s+", line)) {
      # Remove the "+" prefix and one following whitespace if it exists
      clean_line <- sub("^\\+\\s\\s", "", line)
      code_lines <- c(code_lines, clean_line)
    }
  }
}

# Print the extracted code
cat("Extracted code lines:\n")
cat(paste(code_lines, collapse = "\n"), "\n")

# Optionally, write the result to a file
output_file <- "paper/paper_code.R"
# Create the Rmd content
rmd_content <- c(
  "---",
  "title: \"Extracted Code from mlr3torch Paper\"",
  "author: \"mlr3torch\"",
  "date: \"`r Sys.Date()`\"",
  "output: html_document",
  "---",
  "",
  "```{r}",
  code_lines[seq_len(length(code_lines) - 1)],
  "```"
)

code_lines[startsWith(code_lines, "autoplot")] <- paste0("plt <- ", code_lines[startsWith(code_lines, "autoplot")])
code_lines <- c(
  "#' ```{r setup, include=FALSE}",
  "#' knitr::opts_chunk$set(cache = FALSE)",
  "#' ```",
  "Sys.time()",
  "options(mlr3torch.cache = TRUE)",
  "lgr::get_logger(\"mlr3\")$set_threshold(\"warn\")",
  code_lines,
  "print(plt)",
  "saveRDS(plt, \"roc.rds\")",
  "Sys.time()",
  "sessionInfo()"
)


writeLines(code_lines, output_file)
