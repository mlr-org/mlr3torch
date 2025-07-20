library(ggplot2)
library(here)
library(data.table)
library(cowplot)


tbl = readRDS(here::here("paper", "benchmark", "result.rds"))

tbl_med = tbl[, .(time_per_batch_med = median(time_per_batch), loss_med = median(loss)), by = .(n_layers, optimizer, algorithm, jit, latent)]

tbl_cuda = tbl[tag == "cuda_exp", ]
tbl_cpu = tbl[tag == "cpu_exp", ]
tbl_cuda_med = tbl_cuda[, .(time_per_batch_med = median(time_per_batch), loss_med = median(loss)), by = .(n_layers, optimizer, algorithm, jit, latent)]
tbl_cpu_med = tbl_cpu[, .(time_per_batch_med = median(time_per_batch), loss_med = median(loss)), by = .(n_layers, optimizer, algorithm, jit, latent)]

plt <- function(opt_name, cuda) {
  tbl = if (cuda) tbl_cuda_med else tbl_cpu_med

  ggplot(tbl[optimizer == opt_name, ], aes(x = n_layers, y = time_per_batch_med, color = algorithm, linetype = jit)) +
    geom_smooth(method = "lm", se = FALSE) +
    facet_wrap(~latent) +
    labs(
      y = "Time per batch (s)",
      linetype = "JIT",
      color = "Algorithm",
      x = "Number of hidden layers"
    ) +
    theme_bw() +
    theme(legend.position = if (cuda) "top" else "none") +
    ylim(0, NA) +
    scale_linetype_manual(values = c("solid", "twodash"),
                          aesthetics = c("linetype"),
                          guide = guide_legend(override.aes = list(color = "black"))) +
    scale_color_brewer(palette = "Set2") +
    # add label CUDA/CPU on the top left (of the whole plot)
    geom_text(x = -Inf, y = Inf, label = if (cuda) "CUDA" else "CPU", hjust = -0.1, vjust = 1.1, size = 5)
}


plot_cuda_adamw <- plt("adamw", TRUE)
plot_cpu_adamw <- plt("adamw", FALSE)

plot_adamw <- plot_grid(plot_cuda_adamw, plot_cpu_adamw, ncol = 1, rel_heights = c(0.55, 0.45))
plot_adamw

plot_cuda_sgd <- plt("sgd", TRUE)
plot_cpu_sgd <- plt("sgd", FALSE)

plot_cuda_sgd
plot_cuda_adamw
plot_cpu_sgd
plot_cpu_adamw
