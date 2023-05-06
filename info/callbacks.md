The implementation of torch callbacks now differs from the implementation of callbacks in the rest of mlr3.
In general, not all things were considered during the initial design of callbacks. This document lists these issues 
for further references on why things are designed as they are.


The main idea, before the explanation of the design decision:


The class of the callbacks and the parameters are passed to the learner, similar to the optimizer and the loss.


**How to ensure that callbacks are unused**

When passing initialized callbacks to a learner, it is annoying to ensure that the callback has not already been used. 
We would have to write an extra checker function that verifies that all the objects (such as self$progress_bar) are not yet set. 
The alternative is to force all these things to be in self$state, so we can set self$state to NULL before training (or add a check that it must be NULL). 

Argument in favour: It is nice and tidy if everything is in the state
Argument against: This makes the code less readable, because we have to refer to self$state$progress_bar or 
state$progress_bar if lock_objects is FALSE.
If we set lock_objects = FALSE, we can also just refer to progress_bar within the callbacks, otherwise it is self$progress_bar.

Decision: The callbacks should be initialized in the $train() call.

**How the callback receives the context**
We set the context using set_context at the beginning of training.
This fits the style of initializing a context with a given state better (i.e. the decision above).

**Deep Clones**

With the implementation of callbacks in mlr3misc, it was possible to pass initialized callbacks. (As mentioned above, 
the ugly alternative is to store everything in self$state, which I don't like). 
Deep clones of even untrained networks would have to clone those initialized callbacks, which should really not be 
necessary, when initializing the callback in the train function, this means that these deep clones must only create 
a shallow clone of the expressions that we store in the initialize method.

**Parameter and Constructor Arguments**

The learner inherits the paramsets of the callbacks (for that unfortunately the callbacks have to be evaluated, in 
the initialize function to get to the paramset). If we would allow parameter sets and constructor arguments, we would 
have to so



4. In the future, callbacks might have parameters that wa want to tune, for that reason, the callbacks can have parameter sets

5. Callbacks have to work when training an already trained model, otherwise the tests are annoying if we immediately 
throw an error just because we call learner$train(); learner$train(). 


6. The use might cancel the training, in this case the callbacks might be changed and when he then calls train again, 
   stuff might go south. Therefore it really is important, that the callbacks are initialized after the train call.

7. A user might call $continue() (later, when it is available), then the callbacks must not be initialized again but 
the callbacks should be used further (maybe the callback itself should get a flag as to whether it should be re-initialized)

So what has to happen in this case is that the $train() function reconstructs the callbacks. 
For that it is important that we store the information on how to construct the callbacks in the initialize function.

Martin 




Solution:

t_clbk() returns an expression that generates a callback.
This expression is evaluated in the parent environment of the Learner.
The param_set (if available) is added to the param_set of the Learner.


5. We might also want to create callbacks during prediction, e.g. a progress bar for prediction or logging some images and text e.g.
Therefore the callback must also get a field "stage" that indicates whether it is during or prediction.


We write the torch_callback() function so that enforces these things.

1. It is important that the id is set, it therefore has a constructor argument "id".
Usually the class name of the callback can be inferred from the id, so there should be a good default.

2. Because there i



TODO for the Callbacks: 

* Implement a saving callback 
* Implement tflogger callback
* Implement gradient clipping callback

So how do we want the user to interact with that API


```{r}

learner = lrn("classif.mlp", 
  callbacks = list(
    t_clbk("progress", )
    t_clbk("gradient_clipping")
  )


    )

)

t_clbks(progress = list(), saver = list(path = tempfile()))
```

We implement callback stages as private methods so we don't have to document them all explicitely but can still use the R6 class documentation.
