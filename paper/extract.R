# Script to extract R code from paper.tex
# Extracts lines between \begin{CodeInput} and \end{CodeInput}
# Only keeps lines starting with "R>" or "+" and removes these prefixes

# Read the LaTeX file
tex_file <- readLines("paper/paper.tex")

# Initialize variables to track state
in_code_block <- FALSE
code_lines <- character(0)

# Section numbers, used to precede the code with comments that link it to the sections of the paper.
# The comment of a (sub)section is only written before its first code block.
section <- 0L
subsection <- 0L
pending_headers <- character(0)

# Title of a (sub)section, preferring the short title in [...], which has no LaTeX markup
section_title <- function(line) {
  title <- if (grepl("^\\\\(sub)?section\\[", line)) {
    sub("^\\\\(sub)?section\\[([^]]*)\\].*", "\\2", line)
  } else {
    sub("^\\\\(sub)?section\\{([^}]*)\\}.*", "\\2", line)
  }
  gsub("\\\\code\\{([^}]*)\\}", "\\1", title)
}

# Process each line
for (line in tex_file) {
  # Track the (unstarred) sections and subsections
  if (grepl("^\\\\section[[{]", line)) {
    section <- section + 1L
    subsection <- 0L
    pending_headers <- c(sprintf("# %i. %s", section, section_title(line)), "")
  } else if (grepl("^\\\\subsection[[{]", line)) {
    subsection <- subsection + 1L
    # keep the header of the section if none of its code was written yet
    pending_headers <- c(pending_headers[startsWith(pending_headers, sprintf("# %i. ", section)) |
      pending_headers == ""], sprintf("# %i.%i %s", section, subsection, section_title(line)))
  }

  # Check if we're entering a code block
  if (grepl("\\\\begin\\{CodeInput\\}", line)) {
    in_code_block <- TRUE
    code_lines <- c(code_lines, pending_headers)
    pending_headers <- character(0)
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
  "options(width = 70, useFancyQuotes = FALSE)",
  "options(mlr3torch.cache = TRUE)",
  "lgr::get_logger(\"mlr3\")$set_threshold(\"warn\")",
  "",
  code_lines,
  "library(\"ggplot2\")",
  "plt = plt +",
  "  theme(",
  "    axis.text.x = element_text(size = 12),",
  "    axis.text.y = element_text(size = 12),",
  "    axis.title.x = element_text(size = 12),",
  "    axis.title.y = element_text(size = 12)",
  "  )",
  "print(plt)",
  "ggsave(here::here(\"roc.png\"), plt, width = 4, height = 4, dpi = 300)",
  "Sys.time()",
  "sessionInfo()"
)


writeLines(code_lines, output_file)
