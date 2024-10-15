library(here)

library(mlr3oml)
library(data.table)
library(tidytable)

cc18_collection = ocl(99)

cc18_simple = list_oml_data(data_id = cc18_collection$data_ids, 
              number_classes = 2,
              number_missing_values = 0)

cc18_small = cc18_simple |>
  filter(NumberOfSymbolicFeatures == 1) |>
  select(data_id, name, NumberOfFeatures, NumberOfInstances) |>
  filter(name %in% c("qsar-biodeg", "madelon", "kc1", "blood-transfusion-service-center", "climate-model-simulation-crashes"))

# kc1_1067 = odt(1067)


# save the data locally
mlr3misc::pmap(cc18_small, function(data_id, name, NumberOfFeatures, NumberOfInstances) {
  dt = odt(data_id)$data
  dt_name = here("data", "oml", paste0(name, "_", data_id, ".csv"))
  fwrite(dt, file = dt_name)
})

fwrite(cc18_small, here("data", "oml", "cc18_small.csv"))
