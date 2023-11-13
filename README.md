## Image Quality Issues Plugin

![common_image_issues_updated](https://github.com/jacobmarks/image-quality-issues/assets/12500356/4f3b89c9-58b6-4404-a9da-8cd6570a1793)

This plugin is a Python plugin that allows you to find common issues in your
image datasets.

With this plugin, you can find the following issues:

- 📏 Aspect ratio: find images with weird aspect ratios
- 🌫️ Blurriness: find blurry images
- ☀️ Brightness: find bright and dark images
- 🌓 Contrast: find images with high or low contrast
- 🔀 Entropy: find images with low entropy
- 📸 Exposure: find overexposed and underexposed images
- 🕯️ Illumination: find images with uneven illumination
- 🧂 Noise: find images with high salt and pepper noise
- 🌈 Saturation: find images with low and high saturation

### Updates

- **2021-11-13**: Version 2.0.1 supports [calling the compute methods from the Python SDK](#python-sdk)!
- **2021-11-10**: Version 2.0.0 adds support for object patches, lots of refactoring, and more robustness!

## Watch On Youtube

[![Video Thumbnail](https://img.youtube.com/vi/0Kkzx0nEXEo/0.jpg)](https://www.youtube.com/watch?v=0Kkzx0nEXEo&list=PLuREAXoPgT0RZrUaT0UpX_HzwKkoB-S9j&index=14)

## Installation

```shell
fiftyone plugins download https://github.com/jacobmarks/image-quality-issues/
```

## Operators

### `compute_aspect_ratio`

Computes the aspect ratio of all images in the dataset.

### `compute_blurriness`

Computes the blurriness of all images in the dataset.

### `compute_brightness`

Computes the brightness of all images in the dataset.

### `compute_contrast`

Computes the contrast of all images in the dataset.

### `compute_entropy`

Computes the entropy of all images in the dataset.

### `compute_exposure`

Computes the exposure of all images in the dataset.

### `compute_salt_and_pepper`

Computes the salt and pepper noise of all images in the dataset.

### `compute_saturation`

Computes the saturation of all images in the dataset.

### `compute_vignetting`

Computes the vignetting of all images in the dataset.

### `find_issues`

Finds images with brightness, aspect ratio, or entropy issues. You can specify
the threshold for each issue type, and which issue types to check for.

## Python SDK

You can also use the compute operators from the Python SDK!

```python
import fiftyone as fo
import fiftyone.operators as foo
import fiftyone.zoo as foz

dataset = fo.load_dataset("quickstart")

## Access the operator via its URI (plugin name + operator name)
compute_brightness = foo.get_operator(
    "@jacobmarks/image_issues/compute_brightness"
)

## Compute the brightness of all images in the dataset
compute_brightness(dataset)

## Compute brightness of a view
view = dataset.take(10)
compute_brightness(view)

## Compute brightness of detection patches
compute_brightness(dataset, patches_field="ground_truth")
```

**Note**: The `find_issues` operator is not yet supported in the Python SDK.

**Note**: You may have trouble running these within a Jupyter notebook. If so, try running them in a Python script.

## See Also

Here are some related projects that this plugin is inspired by and adapted from:

- [BlurDetection2](https://github.com/WillBrennan/BlurDetection2) with OpenCV
- CleanLab's [CleanVision](https://github.com/cleanlab/cleanvision/tree/main)
- [Exposure_Correction](https://github.com/mahmoudnafifi/Exposure_Correction)
- Lakera's [MLTest](https://docs.lakera.ai/)
