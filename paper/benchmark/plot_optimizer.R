library(ggplot2)
library(data.table)

tbl = readRDS(here::here("paper", "benchmark", "result-linux-gpu-optimizer.rds"))

tbl_summary = tbl[,
    .(time_per_batch_med = median(time_per_batch * 1000),
      time_per_batch_q10 = quantile(time_per_batch * 1000, 0.2),
      time_per_batch_q90 = quantile(time_per_batch * 1000, 0.9)),
    by = .(n_layers, algorithm, jit, latent, optimizer, opt_type)]

ggplot(tbl_summary, aes(x = n_layers, y = time_per_batch_med, color = opt_type)) +
  geom_ribbon(aes(ymin = time_per_batch_q10, ymax = time_per_batch_q90, fill = opt_type), alpha = 0.2, color = NA) +
  geom_line() +
  facet_wrap(vars(optimizer)) +
  geom_point() +
  theme_bw() +
  labs(
    y = "Time per batch (ms)",
    x = "Optimizer"
  )



ggsave(here::here("paper", "benchmark", "plot_optimizer.png"), width = 12, height = 4, dpi = 300)


# ignite relative to standard

tbl_summary_relative = tbl_summary
tbl_summary_relative[, time_per_batch_med_rel := time_per_batch_med / time_per_batch_med[opt_type == "ignite"], by = .(n_layers, optimizer, jit, latent)]
tbl_summary_relative

tbl_summary[opt_type == "standard" & optimizer == "sgd", range(time_per_batch_med_rel)]
tbl_summary[opt_type == "standard" & optimizer == "adamw", range(time_per_batch_med_rel)]


ggplot(tbl_summary_relative, aes(x = n_layers, y = time_per_batch_med_rel, color = opt_type)) +
  geom_line() +
  facet_wrap(vars(optimizer)) +
  geom_point() +
  theme_bw() +
  labs(
    y = "Time per batch (ms)",
    x = "Optimizer"
  )


ggsave(here::here("paper", "benchmark", "plot_optimizer_relative.png"), width = 12, height = 4, dpi = 300)
