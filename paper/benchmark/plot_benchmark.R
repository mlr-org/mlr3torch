library(ggplot2)
library(here)
library(data.table)
library(cowplot)

tbl1 = readRDS(here::here("paper", "benchmark", "result-macos.rds"))
tbl1$os = "macos"
tbl2 = readRDS(here::here("paper", "benchmark", "result-linux-cpu.rds"))
tbl2$os = "linux"
tbl3 = readRDS(here::here("paper", "benchmark", "result-linux-gpu.rds"))
tbl3$os = "linux"

# TODO: Uncertainty bands with upper and lower quantiles

tbl = rbindlist(list(tbl1, tbl2, tbl3))

#tbl_med = tbl[,
#  .(time_per_batch_med = median(time_per_batch), loss_med = median(loss)),
#  by = .(n_layers, optimizer, algorithm, jit, latent, device, os)
#]

tbl_med = tbl

#tbl_med[,
#  time_per_batch_med_rel := (.SD$time_per_batch_med / .SD[algorithm == "pytorch", ]$time_per_batch_med),
#  by = .(n_layers, optimizer, latent, jit, device, os)
#]


tbl_linux_cuda_med = tbl_med[device == "cuda" & os == "linux", ]
tbl_linux_cpu_med = tbl_med[device == "cpu" & os == "linux", ]
tbl_macos_mps_med = tbl_med[device == "mps" & os == "macos", ]
tbl_macos_cpu_med = tbl_med[device == "cpu" & os == "macos", ]

plt <- function(opt_name, gpu, os) {
  tbl = if (os == "linux" && gpu) {
    tbl_linux_cuda_med
  } else if (os == "linux" && !gpu) {
    tbl_linux_cpu_med
  } else if (os == "macos" && gpu) {
    tbl_macos_mps_med
  } else if (os == "macos" && !gpu) {
    tbl_macos_cpu_med
  } else {
    stop()
  }

  ggplot(
    tbl[optimizer == opt_name, ],
    aes(
      x = n_layers,
      y = time_per_batch * 1000,
      color = algorithm,
      linetype = jit
    )
  ) +
    geom_point(size = 0.5) +
    #geom_line() +
    facet_wrap(~latent, scales = "free_y") +
    labs(
      y = "Time per batch (ms)",
      linetype = "JIT",
      color = "Algorithm",
      x = "Number of hidden layers"
    ) +
    theme_bw() +
    theme(legend.position = if (gpu) "bottom" else "none") +
    ylim(0, NA) +
    scale_linetype_manual(
      values = c("solid", "twodash"),
      aesthetics = c("linetype"),
      guide = guide_legend(override.aes = list(color = "black"))
    ) +
    scale_color_brewer(palette = "Set2")
}


plot_cuda_adamw <- plt("adamw", TRUE, "linux") + ggtitle("CUDA")
plot_cpu_adamw <- plt("adamw", FALSE, "linux") + ggtitle("CPU")

plot_adamw <- plot_grid(
  plot_cuda_adamw,
  plot_cpu_adamw,
  ncol = 1,
  rel_heights = c(0.55, 0.45)
)
plot_adamw

plot_cuda_sgd <- plt("sgd", TRUE, "linux") + ggtitle("CUDA")
plot_cpu_sgd <- plt("sgd", FALSE, "linux") + ggtitle("CPU")

plot_sgd <- plot_grid(
  plot_cuda_sgd,
  plot_cpu_sgd,
  ncol = 1,
  rel_heights = c(0.5, 0.5)
)
plot_sgd


plot_macos_adamw_mps <- plt("adamw", TRUE, "macos") + ggtitle("MPS")
plot_macos_adamw_cpu <- plt("adamw", FALSE, "macos") + ggtitle("CPU")

plot_grid(
  plot_macos_adamw_mps,
  plot_macos_adamw_cpu,
  ncol = 1,
  rel_heights = c(0.5, 0.5)
)

plot_macos_sgd_mps <- plt("sgd", TRUE, "macos") + ggtitle("MPS")
plot_macos_sgd_cpu <- plt("sgd", FALSE, "macos") + ggtitle("CPU")

plot_grid(
  plot_macos_sgd_mps,
  plot_macos_sgd_cpu,
  ncol = 1,
  rel_heights = c(0.5, 0.5)
)
