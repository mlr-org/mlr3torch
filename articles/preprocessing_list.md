# Preprocessing & Augmentation

The table below shows all preprocessing and augmentation operations that
are available in `mlr3torch`.

| Key                                                                                                                       | Label                               | Packages    | Feature Types |
|:--------------------------------------------------------------------------------------------------------------------------|:------------------------------------|:------------|:--------------|
| [augment_center_crop](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_augment_center_crop.html)                       | Center Crop Augmentation            | torchvision | lazy_tensor   |
| [augment_color_jitter](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_augment_color_jitter.html)                     | Color Jitter Augmentation           | torchvision | lazy_tensor   |
| [augment_crop](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_augment_crop.html)                                     | Crop Augmentation                   | torchvision | lazy_tensor   |
| [augment_hflip](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_augment_hflip.html)                                   | Horizontal Flip Augmentation        | torchvision | lazy_tensor   |
| [augment_random_affine](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_augment_random_affine.html)                   | Random Affine Augmentation          | torchvision | lazy_tensor   |
| [augment_random_choice](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_augment_random_choice.html)                   | Random Choice Augmentation          | torchvision | lazy_tensor   |
| [augment_random_crop](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_augment_random_crop.html)                       | Random Crop Augmentation            | torchvision | lazy_tensor   |
| [augment_random_horizontal_flip](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_augment_random_horizontal_flip.html) | Random Horizontal Flip Augmentation | torchvision | lazy_tensor   |
| [augment_random_order](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_augment_random_order.html)                     | Random Order Augmentation           | torchvision | lazy_tensor   |
| [augment_random_resized_crop](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_augment_random_resized_crop.html)       | Random Resized Crop Augmentation    | torchvision | lazy_tensor   |
| [augment_random_vertical_flip](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_augment_random_vertical_flip.html)     | Random Vertical Flip Augmentation   | torchvision | lazy_tensor   |
| [augment_resized_crop](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_augment_resized_crop.html)                     | Resized Crop Augmentation           | torchvision | lazy_tensor   |
| [augment_rotate](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_augment_rotate.html)                                 | Rotate Augmentation                 | torchvision | lazy_tensor   |
| [augment_vflip](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_augment_vflip.html)                                   | Vertical Flip Augmentation          | torchvision | lazy_tensor   |
| [trafo_adjust_brightness](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_trafo_adjust_brightness.html)               | Adjust Brightness Transformation    | torchvision | lazy_tensor   |
| [trafo_adjust_gamma](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_trafo_adjust_gamma.html)                         | Adjust Gamma Transformation         | torchvision | lazy_tensor   |
| [trafo_adjust_hue](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_trafo_adjust_hue.html)                             | Adjust Hue Transformation           | torchvision | lazy_tensor   |
| [trafo_adjust_saturation](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_trafo_adjust_saturation.html)               | Adjust Saturation Transformation    | torchvision | lazy_tensor   |
| [trafo_grayscale](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_trafo_grayscale.html)                               | Grayscale Transformation            | torchvision | lazy_tensor   |
| [trafo_nop](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_trafo_nop.html)                                           | No Transformation                   |             | lazy_tensor   |
| [trafo_normalize](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_trafo_normalize.html)                               | Normalization Transformation        | torchvision | lazy_tensor   |
| [trafo_pad](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_trafo_pad.html)                                           | Padding Transformation              | torchvision | lazy_tensor   |
| [trafo_reshape](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_trafo_reshape.html)                                   | Reshaping Transformation            |             | lazy_tensor   |
| [trafo_resize](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_trafo_resize.html)                                     | Resizing Transformation             | torchvision | lazy_tensor   |
| [trafo_rgb_to_grayscale](https://mlr3torch.mlr-org.com/reference/mlr_pipeops_trafo_rgb_to_grayscale.html)                 | RGB to Grayscale Transformation     | torchvision | lazy_tensor   |
