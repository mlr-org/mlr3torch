library(here)
library(mlr3oml)
library(tidytable)

cc18_collection = ocl(99)

cc18_simple = list_oml_data(data_id = cc18_collection$data_ids,
              number_classes = 2,
              number_missing_values = 0)

cc18_small = cc18_simple |>
  filter(NumberOfSymbolicFeatures == 1) |> # the target class is a symbolic feature
  select(data_id, name, NumberOfFeatures, NumberOfInstances) |>
  filter(name %in% c("qsar-biodeg", "madelon", "kc1", "blood-transfusion-service-center", "climate-model-simulation-crashes"))

data_dir =  here("benchmarks", "data")
if (!dir.exists(data_dir)) {
  dir.create(data_dir)
}

options(mlr3oml.cache = here(data_dir, "oml"))
mlr3misc::pwalk(cc18_small, function(data_id, name, NumberOfFeatures, NumberOfInstances) odt(data_id))

dir.create(here("benchmarks", "data", "oml", "collections"))
fwrite(cc18_small, here("benchmarks", "data", "oml", "collections", "cc18_small.csv"))
