
source(here::here("paper/benchmark/time_rtorch.R"))

p = profvis::profvis({
    time_rtorch(
        epochs = 20,
        batch_size = 32,
        n_layers = 8,
        latent = 400,
        n = 1000,
        p = 100,
        device = "cpu",
        jit = FALSE,
        seed = 1,
        mlr3torch = FALSE
    )
})