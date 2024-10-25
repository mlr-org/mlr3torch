library(mlr3torch)
library(here)

library(tidytable)

# TODO: figure out whether we want the v2 file
ground_truth = fread(here::here("cache", "ISIC_2020_Training_GroundTruth.csv"))

ground_truth

# TODO: figure out how to unzip the actual images

# Construct lazy tensor for each image (e.g. a data table with a single ltnsr column)

# Join with the ground truth file
