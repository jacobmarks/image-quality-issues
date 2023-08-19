"""Common Issues plugin.

This plugin provides operators to compute common issues in image datasets.
It is open source, and is adapted heavily from leading open source projects
listed in the README.
|
"""

import numpy as np


import cv2
from PIL import Image, ImageStat


import fiftyone.operators as foo
from fiftyone.operators import types
import fiftyone as fo
from fiftyone import ViewField as F


def get_filepath(sample):
    return (
        sample.local_path if hasattr(sample, "local_path") else sample.filepath
    )


def compute_sample_brightness(sample):
    image = Image.open(get_filepath(sample))
    stat = ImageStat.Stat(image)
    if sample.metadata.num_channels == 3:
        r, g, b = stat.mean
    else:
        mean = stat.mean[0]
        r, g, b = (mean, mean, mean,)

    ## equation from here:
    ## https://www.nbdtech.com/Blog/archive/2008/04/27/calculating-the-perceived-brightness-of-a-color.aspx
    ## and here:
    ## https://github.com/cleanlab/cleanvision/blob/72a1535019fe7b4636d43a9ef4e8e0060b8d66ec/src/cleanvision/issue_managers/image_property.py#L95
    brightness = (
        np.sqrt(0.241 * r**2 + 0.691 * g**2 + 0.068 * b**2) / 255
    )
    return brightness


def compute_dataset_brightness(dataset):
    dataset.add_sample_field("brightness", fo.FloatField)
    for sample in dataset.iter_samples(autosave=True):
        brightness = compute_sample_brightness(sample)
        sample["brightness"] = brightness


class ComputeBrightness(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_brightness",
            label="Common Issues: compute brightness",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.message("compute brightness", label="compute brightness")
        return types.Property(inputs)

    def execute(self, ctx):
        compute_dataset_brightness(ctx.dataset)
        ctx.trigger("reload_dataset")


def compute_sample_aspect_ratio(sample):
    width, height = sample.metadata.width, sample.metadata.height
    ratio = width / height
    return min(ratio, 1 / ratio)


def compute_dataset_aspect_ratio(dataset):
    dataset.compute_metadata()
    dataset.add_sample_field("aspect_ratio", fo.FloatField)
    for sample in dataset.iter_samples(autosave=True):
        aspect_ratio = compute_sample_aspect_ratio(sample)
        sample["aspect_ratio"] = aspect_ratio


class ComputeAspectRatio(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_aspect_ratio",
            label="Common Issues: compute aspect ratio",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.message("compute aspect ratio", label="compute aspect ratio")
        return types.Property(inputs)

    def execute(self, ctx):
        compute_dataset_aspect_ratio(ctx.dataset)
        ctx.trigger("reload_dataset")


def compute_sample_blurriness(sample):
    image = cv2.imread(sample.filepath)
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance


def compute_dataset_blurriness(dataset):
    dataset.add_sample_field("blurriness", fo.FloatField)
    for sample in dataset.iter_samples(autosave=True):
        blurriness = compute_sample_blurriness(sample)
        sample["blurriness"] = blurriness


class ComputeBlurriness(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_blurriness",
            label="Common Issues: compute blurriness",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.message("compute blurriness", label="compute blurriness")
        return types.Property(inputs)

    def execute(self, ctx):
        compute_dataset_blurriness(ctx.dataset)
        ctx.trigger("reload_dataset")


def compute_sample_contrast(sample):
    image = cv2.imread(sample.filepath, cv2.IMREAD_GRAYSCALE)
    # Calculate the histogram
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    min_intensity = np.min(np.where(histogram > 0))
    max_intensity = np.max(np.where(histogram > 0))
    contrast_range = max_intensity - min_intensity
    return contrast_range


def compute_dataset_contrast(dataset):
    dataset.add_sample_field("contrast", fo.FloatField)
    for sample in dataset.iter_samples(autosave=True):
        contrast = compute_sample_contrast(sample)
        sample["contrast"] = contrast


class ComputeContrast(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_contrast",
            label="Common Issues: compute contrast",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.message("compute contrast", label="compute contrast")
        return types.Property(inputs)

    def execute(self, ctx):
        compute_dataset_contrast(ctx.dataset)
        ctx.trigger("reload_dataset")


def compute_sample_saturation(sample):
    image = cv2.imread(sample.filepath)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    return np.mean(saturation)


def compute_dataset_saturation(dataset):
    dataset.add_sample_field("saturation", fo.FloatField)
    for sample in dataset.iter_samples(autosave=True):
        saturation = compute_sample_saturation(sample)
        sample["saturation"] = saturation


class ComputeSaturation(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_saturation",
            label="Common Issues: compute saturation",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.message("compute saturation", label="compute saturation")
        return types.Property(inputs)

    def execute(self, ctx):
        compute_dataset_saturation(ctx.dataset)
        ctx.trigger("reload_dataset")


def compute_sample_entropy(sample):
    image = Image.open(get_filepath(sample))
    return image.entropy()


def compute_dataset_entropy(dataset):
    dataset.add_sample_field("entropy", fo.FloatField)
    for sample in dataset.iter_samples(autosave=True):
        entropy = compute_sample_entropy(sample)
        sample["entropy"] = entropy


class ComputeEntropy(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_entropy",
            label="Common Issues: compute entropy",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.message("compute entropy", label="compute entropy")
        return types.Property(inputs)

    def execute(self, ctx):
        compute_dataset_entropy(ctx.dataset)
        ctx.trigger("reload_dataset")


def compute_sample_exposure(sample):
    image = cv2.imread(sample.filepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    histogram = cv2.calcHist([gray], [0], None, [256], [0,256])
    normalized_histogram = histogram.ravel()/histogram.max()
    return normalized_histogram


def compute_dataset_exposure(dataset):
    dataset.add_sample_field("min_exposure", fo.ArrayField)
    dataset.add_sample_field("max_exposure", fo.ArrayField)
    for sample in dataset.iter_samples(autosave=True):
        exposure = compute_sample_exposure(sample)
        sample["min_exposure"] = exposure[0]
        sample["max_exposure"] = exposure[-1]

    
class ComputeExposure(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="compute_exposure",
            label="Common Issues: compute exposure",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.message("compute exposure", label="compute exposure")
        return types.Property(inputs)

    def execute(self, ctx):
        compute_dataset_exposure(ctx.dataset)
        ctx.trigger("reload_dataset")
        

def _need_to_compute(dataset, field_name):
    if field_name in list(dataset.get_field_schema().keys()):
        return False
    else:
        return field_name not in dataset.first()


def _run_computation(dataset, field_name):
    if field_name == "brightness":
        compute_dataset_brightness(dataset)
    elif field_name == "aspect_ratio":
        compute_dataset_aspect_ratio(dataset)
    elif field_name == "entropy":
        compute_dataset_entropy(dataset)
    elif field_name == "blurriness":
        compute_dataset_blurriness(dataset)
    elif field_name in ["min_exposure", "max_exposure"]:
        compute_dataset_exposure(dataset)
    elif field_name == "contrast":
        compute_dataset_contrast(dataset)
    elif field_name == "saturation":
        compute_dataset_saturation(dataset)
    else:
        raise ValueError("Unknown field name %s" % field_name)


def find_issue_images(
    dataset,
    threshold,
    field_name,
    issue_name,
    lt=True,
):
    dataset.add_sample_field(issue_name, fo.BooleanField)
    if _need_to_compute(dataset, field_name):
        _run_computation(dataset, field_name)

    if lt:
        view = dataset.set_field(issue_name, F(field_name) < threshold)
    else:
        view = dataset.set_field(issue_name, F(field_name) > threshold)
    view.save()
    view = dataset.match(F(issue_name))
    view.tag_samples(issue_name)
    view.tag_samples("issue")
    view.save()


def find_dark_images(dataset, threshold=0.1):
    find_issue_images(dataset, threshold, "brightness", "dark", lt=True)


def find_bright_images(dataset, threshold=0.55):
    find_issue_images(dataset, threshold, "brightness", "bright", lt=False)


def find_weird_aspect_ratio_images(dataset, threshold=0.5):
    find_issue_images(
        dataset, threshold, "aspect_ratio", "weird_aspect_ratio", lt=True
    )


def find_blurry_images(dataset, threshold=100.0):
    find_issue_images(dataset, threshold, "blurriness", "blurry", lt=True)


def find_low_entropy_images(dataset, threshold=5.0):
    find_issue_images(dataset, threshold, "entropy", "low_entropy", lt=True)


def find_low_exposure_images(dataset, threshold=0.1):
    find_issue_images(dataset, threshold, "min_exposure", "low_exposure", lt=True)


def find_high_exposure_images(dataset, threshold=0.7):
    find_issue_images(dataset, threshold, "max_exposure", "high_exposure", lt=False)


def find_low_contrast_images(dataset, threshold=50.):
    find_issue_images(dataset, threshold, "contrast", "low_contrast", lt=True)


def find_high_contrast_images(dataset, threshold=200.):
    find_issue_images(dataset, threshold, "contrast", "high_contrast", lt=False)


def find_low_saturation_images(dataset, threshold=40.):
    find_issue_images(dataset, threshold, "saturation", "low_saturation", lt=True)


def find_high_saturation_images(dataset, threshold=200.):
    find_issue_images(dataset, threshold, "saturation", "high_saturation", lt=False)


class FindIssues(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="find_issues",
            label="Common Issues: find issues",
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()
        form_view = types.View(label="Find Common Issues")
        if ctx.dataset.media_type != "image":
            warning = types.Warning(
                label="This operator is only available for image datasets!"
            )
            inputs.view("warning", warning)
            return types.Property(inputs)

        threshold_view = types.TextFieldView(
            componentsProps={
                "textField": {
                    "step": "0.01",
                    "inputMode": "numeric",
                    "pattern": "[0-9]*",
                },
            }
        )

        #### DARK IMAGES ####
        inputs.bool(
            "dark",
            default=True,
            label="Find dark images in the dataset",
            view=types.CheckboxView(),
        )

        if ctx.params.get("dark", False) == True:
            inputs.float(
                "dark_threshold",
                default=0.1,
                label="darkness threshold",
                view=threshold_view,
            )

        #### BRIGHT IMAGES ####
        inputs.bool(
            "bright",
            default=True,
            label="Find bright images in the dataset",
            view=types.CheckboxView(),
        )

        if ctx.params.get("bright", False) == True:
            inputs.float(
                "bright_threshold",
                default=0.55,
                label="brightness threshold",
                view=threshold_view,
            )

        #### WEIRD ASPECT RATIO IMAGES ####
        inputs.bool(
            "weird_aspect_ratio",
            default=True,
            label="Find weird aspect ratio images in the dataset",
            view=types.CheckboxView(),
        )

        if ctx.params.get("weird_aspect_ratio", False) == True:
            inputs.float(
                "weird_aspect_ratio_threshold",
                default=0.5,
                label="weird aspect ratio threshold",
                view=threshold_view,
            )

        
        #### BLURRY IMAGES ####
        inputs.bool(
            "blurry",
            default=True,
            label="Find blurry images in the dataset",
            view=types.CheckboxView(),
        )

        if ctx.params.get("blurry", False) == True:
            inputs.float(
                "blurry_threshold",
                default=100.0,
                label="blurriness threshold",
                view=threshold_view,
            )


        #### LOW ENTROPY IMAGES ####
        inputs.bool(
            "low_entropy",
            default=True,
            label="Find low entropy images in the dataset",
            view=types.CheckboxView(),
        )

        if ctx.params.get("low_entropy", False) == True:
            inputs.float(
                "low_entropy_threshold",
                default=5.0,
                label="low entropy threshold",
                view=threshold_view,
            )

        
        #### LOW EXPOSURE IMAGES ####
        inputs.bool(
            "low_exposure",
            default=True,
            label="Find low exposure images in the dataset",
            view=types.CheckboxView(),
        )

        if ctx.params.get("low_exposure", False) == True:
            inputs.float(
                "low_exposure_threshold",
                default=0.1,
                label="low exposure threshold",
                view=threshold_view,
            )


        #### HIGH EXPOSURE IMAGES ####
        inputs.bool(
            "high_exposure",
            default=True,
            label="Find high exposure images in the dataset",
            view=types.CheckboxView(),
        )

        if ctx.params.get("high_exposure", False) == True:
            inputs.float(
                "high_exposure_threshold",
                default=0.7,
                label="high exposure threshold",
                view=threshold_view,
            )


        #### LOW CONTRAST IMAGES ####
        inputs.bool(
            "low_contrast",
            default=True,
            label="Find low contrast images in the dataset",
            view=types.CheckboxView(),
        )

        if ctx.params.get("low_contrast", False) == True:
            inputs.float(
                "low_contrast_threshold",
                default=50.,
                label="low contrast threshold",
                view=threshold_view,
            )


        #### HIGH CONTRAST IMAGES ####
        inputs.bool(
            "high_contrast",
            default=True,
            label="Find high contrast images in the dataset",
            view=types.CheckboxView(),
        )

        if ctx.params.get("high_contrast", False) == True:
            inputs.float(
                "high_contrast_threshold",
                default=200.,
                label="high contrast threshold",
                view=threshold_view,
            )


        #### LOW SATURATION IMAGES ####
        inputs.bool(
            "low_saturation",
            default=True,
            label="Find low saturation images in the dataset",
            view=types.CheckboxView(),
        )

        if ctx.params.get("low_saturation", False) == True:
            inputs.float(
                "low_saturation_threshold",
                default=40.,
                label="low saturation threshold",
                view=threshold_view,
            )


        #### HIGH SATURATION IMAGES ####
        inputs.bool(
            "high_saturation",
            default=True,
            label="Find high saturation images in the dataset",
            view=types.CheckboxView(),
        )

        if ctx.params.get("high_saturation", False) == True:
            inputs.float(
                "high_saturation_threshold",
                default=200.,
                label="high saturation threshold",
                view=threshold_view,
            )


        return types.Property(inputs, view=form_view)

    def execute(self, ctx):
        if ctx.params.get("dark", False) == True:
            dark_threshold = ctx.params.get("dark_threshold", 0.1)
            find_dark_images(ctx.dataset, dark_threshold)
        if ctx.params.get("bright", False) == True:
            bright_threshold = ctx.params.get("bright_threshold", 0.55)
            find_bright_images(ctx.dataset, bright_threshold)
        if ctx.params.get("weird_aspect_ratio", False) == True:
            weird_aspect_ratio_threshold = ctx.params.get(
                "weird_aspect_ratio_threshold", 2.5
            )
            find_weird_aspect_ratio_images(
                ctx.dataset, weird_aspect_ratio_threshold
            )
        if ctx.params.get("blurry", False) == True:
            blurry_threshold = ctx.params.get("blurry_threshold", 100.0)
            find_blurry_images(ctx.dataset, blurry_threshold)
        if ctx.params.get("low_entropy", False) == True:
            low_entropy_threshold = ctx.params.get(
                "low_entropy_threshold", 5.0
            )
            find_low_entropy_images(ctx.dataset, low_entropy_threshold)
        if ctx.params.get("low_exposure", False) == True:
            low_exposure_threshold = ctx.params.get(
                "low_exposure_threshold", 0.1
            )
            find_low_exposure_images(ctx.dataset, low_exposure_threshold)
        if ctx.params.get("high_exposure", False) == True:
            high_exposure_threshold = ctx.params.get(
                "high_exposure_threshold", 0.7
            )
            find_high_exposure_images(ctx.dataset, high_exposure_threshold)
        if ctx.params.get("low_contrast", False) == True:
            low_contrast_threshold = ctx.params.get(
                "low_contrast_threshold", 50.
            )
            find_low_contrast_images(ctx.dataset, low_contrast_threshold)
        if ctx.params.get("high_contrast", False) == True:
            high_contrast_threshold = ctx.params.get(
                "high_contrast_threshold", 200.
            )
            find_high_contrast_images(ctx.dataset, high_contrast_threshold)
        if ctx.params.get("low_saturation", False) == True:
            low_saturation_threshold = ctx.params.get(
                "low_saturation_threshold", 40.
            )
            find_low_saturation_images(ctx.dataset, low_saturation_threshold)
        if ctx.params.get("high_saturation", False) == True:
            high_saturation_threshold = ctx.params.get(
                "high_saturation_threshold", 200.
            )
            find_high_saturation_images(ctx.dataset, high_saturation_threshold)
        ctx.trigger("reload_dataset")


def register(plugin):
    plugin.register(ComputeAspectRatio)
    plugin.register(ComputeBlurriness)
    plugin.register(ComputeBrightness)
    plugin.register(ComputeContrast)
    plugin.register(ComputeEntropy)
    plugin.register(ComputeExposure)
    plugin.register(ComputeSaturation)
    plugin.register(FindIssues)
