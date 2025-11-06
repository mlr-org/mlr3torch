# Melanoma Image classification

Classification of melanoma tumor images. The data is a preprocessed
version of the 2020 SIIM-ISIC challenge where the images have been
reshaped to size \$(3, 128, 128)\$.

By default only the training rows are active in the task, but the test
data (that has no targets) is also included. Whether an observation is
part of the train or test set is indicated by the column `"test"`.

There are no labels for the test rows, so by default, these observations
are inactive, which means that the task uses only 32701 of the 43683
observations that are defined in the underlying data backend.

The data backend also contains a more detailed `diagnosis` of the
specific type of tumor.

Columns:

- `outcome` (factor): the target variable. Whether the tumor is benign
  or malignant (the positive class)

- `anatom_site_general_challenge` (factor): the location of the tumor on
  the patient's body

- `sex` (factor): the sex of the patient

- `age_approx` (int): approximate age of the patient at the time of
  imaging

- `image` (lazy_tensor): The image (shape \$(3, 128, 128)\$) of the
  tumor. ee `split` (character): Whether the observation os part of the
  train or test set.

## Source

<https://huggingface.co/datasets/carsonzhang/ISIC_2020_small>

## Construction

    tsk("melanoma")

## Download

The [task](https://mlr3.mlr-org.com/reference/Task.html)'s backend is a
[`DataBackendLazy`](https://mlr3torch.mlr-org.com/reference/mlr_backends_lazy.md)
which will download the data once it is requested. Other meta-data is
already available before that. You can cache these datasets by setting
the `mlr3torch.cache` option to `TRUE` or to a specific path to be used
as the cache directory.

## Properties

- Task type: “classif”

- Properties: “twoclass”, “groups”

- Has Missings: no

- Target: “outcome”

- Features: “sex”, “anatom_site_general_challenge”, “age_approx”,
  “image”

- Data Dimension: 43683x11

## References

Rotemberg, V., Kurtansky, N., Betz-Stablein, B., Caffery, L., Chousakos,
E., Codella, N., Combalia, M., Dusza, S., Guitera, P., Gutman, D.,
Halpern, A., Helba, B., Kittler, H., Kose, K., Langer, S., Lioprys, K.,
Malvehy, J., Musthaq, S., Nanda, J., Reiter, O., Shih, G., Stratigos,
A., Tschandl, P., Weber, J., Soyer, P. (2021). “A patient-centric
dataset of images and metadata for identifying melanomas using clinical
context.” *Scientific Data*, **8**, 34.
[doi:10.1038/s41597-021-00815-z](https://doi.org/10.1038/s41597-021-00815-z)
.

## Examples

``` r
task = tsk("melanoma")
```
