library(data.table)
library(mlr3)

library(here)

bmr_ce = fread(here("benchmarks", "rf_use_case", "results", "bmr_ce.csv"))

bmr_ce

time = fread(here("benchmarks", "rf_use_case", "results", "time.csv"))

time
