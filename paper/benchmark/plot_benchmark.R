library(ggplot2)
library(here)
library(data.table)
library(cowplot)


tbl = readRDS(here::here("paper", "benchmark", "result.rds"))

tbl_med = tbl[,
  .(time_per_batch_med = median(time_per_batch), loss_med = median(loss)),
  by = .(n_layers, optimizer, algorithm, jit, latent, device)
]

tbl_med[,
  time_per_batch_med_rel := (.SD$time_per_batch_med / .SD[algorithm == "pytorch", ]$time_per_batch_med),
  by = .(n_layers, optimizer, latent, jit, device)
]

tbl_cuda_med = tbl_med[device == "cuda", ]
tbl_cpu_med = tbl_med[device == "cpu", ]

plt <- function(opt_name, cuda) {
  tbl = if (cuda) tbl_cuda_med else tbl_cpu_med

  ggplot(
    tbl[optimizer == opt_name, ],
    aes(
      x = n_layers,
      y = time_per_batch_med * 1000,
      color = algorithm,
      linetype = jit
    )
  ) +
    geom_point(size = 0.5) +
    geom_line() +
    facet_wrap(~latent, scales = "free_y") +
    labs(
      y = "Time per batch (ms)",
      linetype = "JIT",
      color = "Algorithm",
      x = "Number of hidden layers"
    ) +
    theme_bw() +
    theme(legend.position = if (cuda) "bottom" else "none") +
    ylim(0, NA) +
    scale_linetype_manual(
      values = c("solid", "twodash"),
      aesthetics = c("linetype"),
      guide = guide_legend(override.aes = list(color = "black"))
    ) +
    scale_color_brewer(palette = "Set2")
}


plot_cuda_adamw <- plt("adamw", TRUE) + ggtitle("CUDA")
plot_cpu_adamw <- plt("adamw", FALSE) + ggtitle("CPU")

plot_adamw <- plot_grid(
  plot_cuda_adamw,
  plot_cpu_adamw,
  ncol = 1,
  rel_heights = c(0.55, 0.45)
)
plot_adamw

plot_cuda_sgd <- plt("sgd", TRUE) + ggtitle("CUDA")
plot_cpu_sgd <- plt("sgd", FALSE) + ggtitle("CPU")

plot_grid(
  plot_cpu_sgd,
  plot_cpu_adamw,
  ncol = 1
)

plot_sgd <- plot_grid(
  plot_cuda_sgd,
  plot_cpu_sgd,
  ncol = 1,
  rel_heights = c(0.5, 0.5)
)
plot_sgd
