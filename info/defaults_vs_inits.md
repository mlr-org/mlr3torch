# Initial Values vs. Defaults

In various `mlr3` packages there is some confusion between *initial values* and defaults.
A default value is a value that is implicitly used by an upstream function when no **value is set** for a specific parameter.
A initial value is the value to which a parameter is **set** during the construction.

The relevant question here is when to use what and how to do so:

* defaults are used when we call an upstream function. In this case the default in the documentation should also be mentioned along the lines of "... Default is x."
* Initial values should almost be used when mlr3torch implements some logic, as this avoids annoying code snippets like `x = param_vals$x %??% 1L` all over the place.
  A possible exception is when there is a function defined in mlr3torch that has default values and this function is called in a `PipeOp` or `Learner`, then it also makes sense to use defaults.
  The documentation here should be something along the lines of "... Is initialized to x."
