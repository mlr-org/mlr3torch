test_that("autotest", {
    cb = t_clbk("tb")
    expect_torch_callback(cb)
})

# TODO: investigate what's happening when there is only a single epoch (why don't we log anything?)
test_that("a simple example works", {
    # using a temp dir
    cb = t_clbk("tb")

    # check that directory doesn't exist
    expect_false(dir.exists(cb$path))

    # check that the correct training measure name was logged at the correct time (correct epoch)
    task = tsk("iris")

    n_epochs = 10
    batch_size = 50
    neurons = 200
    mlp = lrn("classif.mlp", 
          callbacks = cb,
          epochs = n_epochs, batch_size = batch_size, neurons = neurons,
          validate = 0.2, 
          measures_valid = msrs(c("classif.acc", "classif.ce")), 
          measures_train = msrs(c("classif.acc", "classif.ce"))
    )

    mlp$train(task)

    events = collect_events(cb$path)$summary %>%
        mlr3misc::map(unlist)

    # TODO: this but for the validation measures
    n_last_loss = mlr3misc::map(\(x) x["tag"] == "last_loss") %>%
        unlist() %>%
        sum()
    expect_equal(n_last_loss, n_epochs)
    
    # check that logging happens at the same frequency as eval_freq
})