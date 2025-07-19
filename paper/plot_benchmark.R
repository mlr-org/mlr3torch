library(ggplot2)
library(here)
library(data.table)


tbl = readRDS(here::here("paper", "benchmark", "result.rds"))

tbl_cuda = tbl[tag == "cuda_exp", ]
tbl_cpu = tbl[tag == "cpu_exp", ]


ggplot(tbl_cuda[optimizer == "adamw", ], aes(x = n_layers, y = loss, color = algorithm, linetype = jit)) +
  geom_point() +
  geom_smooth(se = TRUE) +
  facet_wrap(~latent) +
  labs(
    y = "Time per batch (s)",
  )
  theme_bw()

ggplot(tbl_cpu) +
  geom_point(aes(x = n_epochs, y = time, color = model)) +
  geom_line(aes(x = n_epochs, y = time, color = model)) +
  facet_wrap(~model)
