library(ggplot2)
library(here)

# Just postproceses the ROC curve plot for better readability

plt = readRDS(here("paper_results", "roc.rds"))
# increase the size of x axis and x labels and y axis and y labels

plt = plt +
  theme(
    axis.text.x = element_text(size = 12),
    axis.text.y = element_text(size = 12),
    axis.title.x = element_text(size = 12),
    axis.title.y = element_text(size = 12)
  )


ggsave(here("roc.png"), plt, width = 4, height = 4, dpi = 300)
