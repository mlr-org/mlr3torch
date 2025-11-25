library(ggplot2)
library(here)
library(data.table)
library(cowplot)

tbl1 = readRDS(here::here("paper", "benchmark", "result-linux-cpu.rds"))
tbl1$os = "linux"
tbl2 = readRDS(here::here("paper", "benchmark", "result-linux-gpu.rds"))
tbl2$os = "linux"

TEXT_SIZE = 12

tbl = rbindlist(list(tbl1, tbl2))

tbl_linux_cuda = tbl[device == "cuda" & os == "linux", ]
tbl_linux_cpu = tbl[device == "cpu" & os == "linux", ]

plt <- function(opt_name, gpu, os, show_y_label = TRUE, show_x_label = TRUE) {
  tbl = if (os == "linux" && gpu) {
    tbl_linux_cuda
  } else if (os == "linux" && !gpu) {
    tbl_linux_cpu
  } else {
    stop()
  }

  # Calculate quantiles for each group
  tbl_summary = tbl[optimizer == opt_name,
    .(time_per_batch_med = median(time_per_batch * 1000),
      time_per_batch_q10 = quantile(time_per_batch * 1000, 0.2),
      time_per_batch_q90 = quantile(time_per_batch * 1000, 0.9)),
    by = .(n_layers, algorithm, latent)]

  ggplot(
    tbl_summary,
    aes(
      x = n_layers,
      y = time_per_batch_med,
      color = algorithm
    )
  ) +
    geom_ribbon(
      aes(ymin = time_per_batch_q10, ymax = time_per_batch_q90, fill = algorithm),
      alpha = 0.2,
      color = NA
    ) +
    geom_line() +
    geom_point(size = 0.8) +
    facet_wrap(~latent, scales = "free_y") +
    labs(
      y = if (show_y_label) "Time per batch (ms)" else NULL,
      color = "Algorithm",
      x = if (show_x_label) "Number of hidden layers" else NULL
    ) +
    theme_bw() +
    theme(legend.position = if (!gpu) "left" else "none") +
    ylim(0, NA) +
    scale_linetype_manual(
      values = c("solid", "twodash"),
      aesthetics = c("linetype"),
      guide = guide_legend(override.aes = list(color = "black"))
    ) +
    scale_color_manual(
      values = c("rtorch" = "#66C2A5", "mlr3torch" = "#FC8D62", pytorch = "#8DA0CB")
    ) +
    scale_fill_manual(
      values = c("rtorch" = "#66C2A5", "mlr3torch" = "#FC8D62", pytorch = "#8DA0CB")
    )
}


plot_cuda_adamw <- plt("adamw", TRUE, "linux", show_x_label = FALSE) + ggtitle("AdamW / CUDA") + theme(plot.title = element_text(size = TEXT_SIZE))
plot_cuda_sgd <- plt("sgd", TRUE, "linux", show_y_label = FALSE, show_x_label = FALSE) + ggtitle("SGD / CUDA") + theme(plot.title = element_text(size = TEXT_SIZE))

plot_cuda <- plot_grid(
  plot_cuda_adamw,
  plot_cuda_sgd,
  ncol = 2
)

plot_cpu_adamw <- plt("adamw", FALSE, "linux") + ggtitle("AdamW / CPU") + theme(legend.position = "none", plot.title = element_text(size = TEXT_SIZE))
plot_cpu_sgd <- plt("sgd", FALSE, "linux", show_y_label = FALSE) + ggtitle("SGD / CPU") + theme(plot.title = element_text(size = TEXT_SIZE)) + theme(legend.position = "none")
plot_legend = plt("adamw", FALSE, "linux") + theme(legend.position = "bottom")


legend = get_legend(plot_legend)

plot_cpu <- plot_grid(
  plot_cpu_adamw,
  plot_cpu_sgd,
  ncol = 2
)

plot_combined <- plot_grid(
  plot_cuda,
  legend,
  plot_cpu,
  ncol = 1,
  rel_heights = c(1, 0.1, 1)
)

ggsave(here::here("paper", "benchmark", "plot_benchmark.png"),
  plot_combined, width = 12, height = 4, dpi = 300)

plt_relative <- function(opt_name, gpu, os, show_y_label = TRUE, show_x_label = TRUE) {
  tbl = if (os == "linux" && gpu) {
    tbl_linux_cuda
  } else if (os == "linux" && !gpu) {
    tbl_linux_cpu
  } else {
    stop()
  }

  # Calculate quantiles for each group
  tbl_summary = tbl[optimizer == opt_name,
    .(time_per_batch_med = median(time_per_batch * 1000)),
    by = .(n_layers, algorithm, latent)]


  # not relative to pytorch
  tbl_summary[, time_per_batch_med_rel := time_per_batch_med / time_per_batch_med[algorithm == "pytorch"], by = .(n_layers, latent)]
  tbl_summary = tbl_summary[algorithm != "pytorch", ]

  ggplot(
    tbl_summary,
    aes(
      x = n_layers,
      y = time_per_batch_med_rel,
      color = algorithm
    )
  ) +
    geom_line() +
    geom_point(size = 0.8) +
    facet_wrap(~latent) +
    labs(
      y = if (show_y_label) "Time per batch (ms)" else NULL,
      color = "Algorithm",
      x = if (show_x_label) "Number of hidden layers" else NULL
    ) +
    theme_bw() +
    theme(legend.position = if (!gpu) "left" else "none") +
    ylim(0, NA) +
    scale_color_manual(
      values = c("rtorch" = "#66C2A5", "mlr3torch" = "#FC8D62")
    )
}
plot_cuda_adamw <- plt_relative("adamw", TRUE, "linux", show_x_label = FALSE) + ggtitle("AdamW / CUDA") + theme(plot.title = element_text(size = TEXT_SIZE)) +
  labs(
    y = "Relative median time\n per batch"
  ) + geom_hline(yintercept = 1, linetype = "dashed", alpha = 0.9, color = "grey")
plot_cuda_adamw
plot_cuda_sgd <- plt_relative("sgd", TRUE, "linux", show_y_label = FALSE, show_x_label = FALSE) + ggtitle("SGD / CUDA") + theme(plot.title = element_text(size = TEXT_SIZE)) + geom_hline(yintercept = 1, linetype = "dashed", alpha = 0.9, color = "grey")

plot_cuda <- plot_grid(
  plot_cuda_adamw,
  plot_cuda_sgd,
  ncol = 2
)

plot_cpu_adamw <- plt_relative("adamw", FALSE, "linux") + ggtitle("AdamW / CPU") + theme(legend.position = "none", plot.title = element_text(size = TEXT_SIZE)) +
  labs(
    y = "Relative median time\n per batch"
  ) + geom_hline(yintercept = 1, linetype = "dashed", alpha = 0.9, color = "grey")
plot_cpu_sgd <- plt_relative("sgd", FALSE, "linux", show_y_label = FALSE) + ggtitle("SGD / CPU") + theme(plot.title = element_text(size = TEXT_SIZE)) + theme(legend.position = "none") + geom_hline(yintercept = 1, linetype = "dashed", alpha = 0.9, color = "grey")
plot_legend = plt_relative("adamw", FALSE, "linux") + theme(legend.position = "bottom")


legend = get_legend(plot_legend)

plot_cpu <- plot_grid(
  plot_cpu_adamw,
  plot_cpu_sgd,
  ncol = 2
)

plot_combined <- plot_grid(
  plot_cuda,
  legend,
  plot_cpu,
  ncol = 1,
  rel_heights = c(1, 0.1, 1)
)
plot_combined

ggsave(here::here("paper", "benchmark", "plot_benchmark_relative.png"),
  plot_combined, width = 12, height = 4, dpi = 300)

tbl_linux_cuda[optimizer == "adamw", mean(median(time_per_batch * 1000))]
