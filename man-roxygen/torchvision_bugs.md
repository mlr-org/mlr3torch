# torchvision bugs found while implementing the shape inference

Six deviations of the R `torchvision` package from the `torchvision.transforms.functional`
behaviour of PyTorch. The first four were found by comparing the inferred output shapes against
the shapes the functions actually return, the last two while writing tests for the hue
transformation. Every snippet below runs against `torchvision` 0.7.0 / `torch` 0.17.0.

The `FIXME(torchvision)` comments in `R/preprocess.R` point here. The first two configurations are
rejected by `mlr3torch` until they are fixed upstream; bugs 3 and 4 are followed as they are,
because fixing them upstream changes the output shape and therefore our inference as well.
Bugs 5 and 6 do not affect shapes, so they need nothing on our side.

All six are fixed on the `fix/transform-shape-bugs` branch of the local `torchvision` checkout.

```r
library(torch)
library(torchvision)
img = torch_rand(3, 16, 20)   # (channels, height, width)
```

## 1. `transform_center_crop()` pads a non-square `size` incorrectly

When the image is smaller than the requested size, the function pads it. In the padding branch
height and width are used in the wrong order, so the result is not the requested size and can even
be empty.

```r
dim(transform_center_crop(img, c(24, 30)))   # 3 22  0   -- expected 3 24 30
dim(transform_center_crop(img, c(24, 10)))   # 3  0 10   -- expected 3 24 10
dim(transform_center_crop(img, c(8, 30)))    # 3  6  0   -- expected 3  8 30
dim(transform_center_crop(img, 32))          # 3 32 32   -- correct, square size
dim(transform_center_crop(img, c(8, 10)))    # 3  8 10   -- correct, no padding needed
```

Only the padded, non-square case is affected. PyTorch returns the requested size in all of them.

## 2. `transform_crop()` clamps instead of padding

`F.crop()` in PyTorch pads the result with zeros when the crop leaves the image, so the output
always has the requested size. The R version returns whatever part of the image the crop covers.

```r
dim(transform_crop(img, top = 10, left = 10, height = 20, width = 20))  # 3  7 11 -- expected 3 20 20
dim(transform_crop(img, top = 1, left = 1, height = 20, width = 24))    # 3 16 20 -- expected 3 20 24
```

A crop that lies entirely outside the image gives an extent of 0, i.e. a tensor with no elements.

## 3. `transform_rgb_to_grayscale()` drops the channel dimension

PyTorch returns a tensor with one channel, i.e. the rank is preserved. The R version removes the
dimension, which changes the rank and breaks any operator downstream that indexes dimensions by
position.

```r
dim(transform_rgb_to_grayscale(img))                    # 16 20     -- expected 1 16 20
dim(transform_rgb_to_grayscale(torch_rand(2, 3, 16, 20)))  # 2 16 20 -- expected 2 1 16 20
```

Note that `transform_grayscale(img, num_output_channels = 1)` does the right thing and returns
`1 16 20`, so the two functions disagree with each other.

## 4. `transform_adjust_hue()` silently drops channels

PyTorch requires an image with 1 or 3 channels and raises otherwise. The R version truncates
silently, so a 4-channel image loses its fourth channel without any indication.

```r
dim(transform_adjust_hue(torch_rand(4, 16, 20), 0.2))   # 3 16 20 -- expected an error
```

`transform_color_jitter()` inherits this via its `hue` argument.

## 5. `hsv2rgb()` does not invert `rgb2hsv()`

The fractional part of the hue sector is computed as `h * 6 - 1` instead of `h * 6 - floor(h * 6)`,
and the `t` component omits the saturation, so converting an image to HSV and back does not return
it. `transform_adjust_hue()` and `transform_color_jitter(hue = ...)` are built on this.

```r
x = torch_rand(3, 8, 10)
max(abs(as.array(x) - as.array(torchvision:::hsv2rgb(torchvision:::rgb2hsv(x)))))  # 0.68
dim(transform_adjust_hue(x, 0))                                    # unchanged shape
max(abs(as.array(x) - as.array(transform_adjust_hue(x, 0))))       # 0.68 -- expected 0
```

## 6. `transform_adjust_hue()` gives a black image for a negative `hue_factor`

The hue is wrapped with `%%`, which is `fmod()` for tensors and keeps the sign of the dividend, so
a negative factor leaves the hue negative and the conversion back to RGB yields zeros. Python's `%`
wraps into `[0, 1)`.

```r
red = torch_tensor(array(c(1, 0, 0), dim = c(3, 1, 1)))
as.numeric(transform_adjust_hue(red, 1 / 3))    # 0 1 0  -- correct, green
as.numeric(transform_adjust_hue(red, -1 / 3))   # 0 0 0  -- expected 0 0 1, blue
```
