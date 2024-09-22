test_that("autotest", {
    cb = t_clbk("tb")
    expect_torch_callback(cb)
})

# TODO: investigate what's happening when there is only a single epoch (why don't we log anything?)
test_that("a simple example works", {
    # using a temp dir

    # check that directory doesn't exist

    # check that directory was created

    # check that default logging directory is the directory name we passed in

    # check that the correct training measure name was logged at the correct time (correct epoch)

    # check that the correct validation measure name was logged

    # check that logging happens at the same frequency as eval_freq
})