Whenever the mlr3torch package implements a parameter we initialize the parameter value.
If the parameter is implemented by another package, we use a default value.

This ensures that when the value is accessed during train / predict and the parameter is processed by mlr3torch
functions, we do not have to do stuff like `x = param_vals$x %??% 1L` all over the place.
