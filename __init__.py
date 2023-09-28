"""Common Issues plugin.

This plugin provides operators to compute common issues in image datasets.
It is open source, and is adapted heavily from leading open source projects
listed in the README.
|
"""

import numpy as np


import cv2
from PIL import Image


import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types
from fiftyone import ViewField as F


def _execution_mode(ctx, inputs):
    delegate = ctx.params.get("delegate", False)

    if delegate:
        description = "Uncheck this box to execute the operation immediately"
    else:
        description = "Check this box to delegate execution of this task"

    inputs.bool(
        "delegate",
        default=False,
        required=True,
        label="Delegate execution?",
        description=description,
        view=types.CheckboxView(),
    )

    if delegate:
        inputs.view(
            "notice",
            types.Notice(
                label=(
                    "You've chosen delegated execution. Note that you must "
                    "have a delegated operation service running in order for "
                    "this task to be processed. See "
                    "https://docs.voxel51.com/plugins/index.html#operators "
                    "for more information"
                )
            ),
        )


def _list_target_views(ctx, inputs):
    has_view = ctx.view != ctx.dataset.view()
    has_selected = bool(ctx.selected)
    default_target = "DATASET"
    if has_view or has_selected:
        target_choices = types.RadioGroup()
        target_choices.add_choice(
            "DATASET",
            label="Entire dataset",
            description="Run computation for the entire dataset",
        )

        if has_view:
            target_choices.add_choice(
                "CURRENT_VIEW",
                label="Current view",
                description="Run computation for the current view",
            )
            default_target = "CURRENT_VIEW"

        if has_selected:
            target_choices.add_choice(
                "SELECTED_SAMPLES",
                label="Selected samples",
                description="Run computation for the selected samples",
            )
            default_target = "SELECTED_SAMPLES"

        inputs.enum(
            "target",
            target_choices.values(),
            default=default_target,
            view=target_choices,
        )
    else:
        ctx.params["target"] = "DATASET"


def _get_target_view(ctx, target):
    if target == "SELECTED_SAMPLES":
        return ctx.view.select(ctx.selected)

    if target == "DATASET":
        return ctx.dataset

    return ctx.view


def get_filepath(sample):
    return (
        sample.local_path if hasattr(sample, "local_path") else sample.filepath
    )


def compute_sample_brightness(sample):
    image = Image.open(get_filepath(sample))
    pixels = np.array(image)
    if sample.metadata.num_channels == 3:
        r, g, b = pixels.mean(axis=(0, 1))
    else:
        mean = pixels.mean()
        r, g, b = (
            mean,
            mean,
            mean,
        )

    ## equation from here:
    ## https://www.nbdtech.com/Blog/archive/2008/04/27/calculating-the-perceived-brightness-of-a-color.aspx
    ## and here:
    ## https://github.com/cleanlab/cleanvision/blob/72a1535019fe7b4636d43a9ef4e8e0060b8d66ec/src/cleanvision/issue_managers/image_property.py#L95
    brightness = (
        np.sqrt(0.241 * r**2 + 0.691 * g**2 + 0.068 * b**2) / 255
    )
    return brightness


def compute_dataset_brightness(dataset, view=None):
    dataset.add_sample_field("brightness", fo.FloatField)
    if view is None:
        view = dataset
    for sample in view.iter_samples(autosave=True):
        brightness = compute_sample_brightness(sample)
        sample["brightness"] = brightness


class ComputeBrightness(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="compute_brightness",
            label="Common Issues: compute brightness",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.message("compute brightness", label="compute brightness")
        _execution_mode(ctx, inputs)
        _list_target_views(ctx, inputs)
        return types.Property(inputs)

    def execute(self, ctx):
        view = _get_target_view(ctx, ctx.params["target"])
        compute_dataset_brightness(ctx.dataset, view=view)
        ctx.trigger("reload_dataset")


def compute_sample_aspect_ratio(sample):
    width, height = sample.metadata.width, sample.metadata.height
    ratio = width / height
    return min(ratio, 1 / ratio)


def compute_dataset_aspect_ratio(dataset, view=None):
    dataset.compute_metadata()
    dataset.add_sample_field("aspect_ratio", fo.FloatField)
    if view is None:
        view = dataset
    for sample in view.iter_samples(autosave=True):
        aspect_ratio = compute_sample_aspect_ratio(sample)
        sample["aspect_ratio"] = aspect_ratio


class ComputeAspectRatio(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="compute_aspect_ratio",
            label="Common Issues: compute aspect ratio",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.message("compute aspect ratio", label="compute aspect ratio")
        _execution_mode(ctx, inputs)
        _list_target_views(ctx, inputs)
        return types.Property(inputs)

    def execute(self, ctx):
        view = _get_target_view(ctx, ctx.params["target"])
        compute_dataset_aspect_ratio(ctx.dataset, view=view)
        ctx.trigger("reload_dataset")


def compute_sample_blurriness(sample):
    # pylint: disable=no-member
    image = cv2.imread(get_filepath(sample))
    # Convert the image to grayscale
    # pylint: disable=no-member
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # pylint: disable=no-member
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance


def compute_dataset_blurriness(dataset, view=None):
    dataset.add_sample_field("blurriness", fo.FloatField)
    if view is None:
        view = dataset
    for sample in view.iter_samples(autosave=True):
        blurriness = compute_sample_blurriness(sample)
        sample["blurriness"] = blurriness


class ComputeBlurriness(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="compute_blurriness",
            label="Common Issues: compute blurriness",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.message("compute blurriness", label="compute blurriness")
        _execution_mode(ctx, inputs)
        _list_target_views(ctx, inputs)
        return types.Property(inputs)

    def execute(self, ctx):
        view = _get_target_view(ctx, ctx.params["target"])
        compute_dataset_blurriness(ctx.dataset, view=view)
        ctx.trigger("reload_dataset")


def compute_sample_contrast(sample):
    # pylint: disable=no-member
    image = cv2.imread(get_filepath(sample), cv2.IMREAD_GRAYSCALE)

    # Calculate the histogram
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    min_intensity = np.min(np.where(histogram > 0))
    max_intensity = np.max(np.where(histogram > 0))
    contrast_range = max_intensity - min_intensity
    return contrast_range


def compute_dataset_contrast(dataset, view=None):
    dataset.add_sample_field("contrast", fo.FloatField)
    if view is None:
        view = dataset
    for sample in view.iter_samples(autosave=True):
        contrast = compute_sample_contrast(sample)
        sample["contrast"] = contrast


class ComputeContrast(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="compute_contrast",
            label="Common Issues: compute contrast",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.message("compute contrast", label="compute contrast")
        _execution_mode(ctx, inputs)
        _list_target_views(ctx, inputs)
        return types.Property(inputs)

    def execute(self, ctx):
        view = _get_target_view(ctx, ctx.params["target"])
        compute_dataset_contrast(ctx.dataset, view=view)
        ctx.trigger("reload_dataset")


def compute_sample_saturation(sample):
    # pylint: disable=no-member
    image = cv2.imread(get_filepath(sample))
    # pylint: disable=no-member
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    return np.mean(saturation)


def compute_dataset_saturation(dataset, view=None):
    dataset.add_sample_field("saturation", fo.FloatField)
    if view is None:
        view = dataset
    for sample in view.iter_samples(autosave=True):
        saturation = compute_sample_saturation(sample)
        sample["saturation"] = saturation


class ComputeSaturation(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="compute_saturation",
            label="Common Issues: compute saturation",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.message("compute saturation", label="compute saturation")
        _execution_mode(ctx, inputs)
        _list_target_views(ctx, inputs)
        return types.Property(inputs)

    def execute(self, ctx):
        view = _get_target_view(ctx, ctx.params["target"])
        compute_dataset_saturation(ctx.dataset, view=view)
        ctx.trigger("reload_dataset")


def compute_sample_entropy(sample):
    image = Image.open(get_filepath(sample))
    return image.entropy()


def compute_dataset_entropy(dataset, view=None):
    dataset.add_sample_field("entropy", fo.FloatField)
    if view is None:
        view = dataset
    for sample in view.iter_samples(autosave=True):
        entropy = compute_sample_entropy(sample)
        sample["entropy"] = entropy


class ComputeEntropy(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="compute_entropy",
            label="Common Issues: compute entropy",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.message("compute entropy", label="compute entropy")
        _execution_mode(ctx, inputs)
        _list_target_views(ctx, inputs)
        return types.Property(inputs)

    def execute(self, ctx):
        view = _get_target_view(ctx, ctx.params["target"])
        compute_dataset_entropy(ctx.dataset, view=view)
        ctx.trigger("reload_dataset")


def compute_sample_exposure(sample):
    # pylint: disable=no-member
    image = cv2.imread(get_filepath(sample))
    # pylint: disable=no-member
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # pylint: disable=no-member
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
    normalized_histogram = histogram.ravel() / histogram.max()
    return normalized_histogram


def compute_dataset_exposure(dataset, view=None):
    dataset.add_sample_field("min_exposure", fo.FloatField)
    dataset.add_sample_field("max_exposure", fo.FloatField)
    if view is None:
        view = dataset
    for sample in view.iter_samples(autosave=True):
        exposure = compute_sample_exposure(sample)
        sample["min_exposure"] = exposure[0]
        sample["max_exposure"] = exposure[-1]


class ComputeExposure(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="compute_exposure",
            label="Common Issues: compute exposure",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.message("compute exposure", label="compute exposure")
        _execution_mode(ctx, inputs)
        _list_target_views(ctx, inputs)
        return types.Property(inputs)

    def execute(self, ctx):
        view = _get_target_view(ctx, ctx.params["target"])
        compute_dataset_exposure(ctx.dataset, view=view)
        ctx.trigger("reload_dataset")


def compute_sample_salt_and_pepper(sample):
    """
    Computes the salt and pepper noise of an image.
    """
    SALT_THRESHOLD = 245
    PEPPER_THRESHOLD = 10

    # pylint: disable=no-member
    image = cv2.imread(get_filepath(sample), cv2.IMREAD_GRAYSCALE)

    # Identify salt-and-pepper pixels
    salt_pixels = image >= SALT_THRESHOLD
    pepper_pixels = image <= PEPPER_THRESHOLD

    # Calculate the percentage of salt-and-pepper pixels
    total_salt_pepper_pixels = np.sum(salt_pixels) + np.sum(pepper_pixels)
    total_pixels = image.size
    noise_percentage = total_salt_pepper_pixels / total_pixels * 100
    return noise_percentage


def compute_dataset_salt_and_pepper(dataset, view=None):
    dataset.add_sample_field("salt_and_pepper", fo.FloatField)
    if view is None:
        view = dataset
    for sample in view.iter_samples(autosave=True):
        salt_and_pepper = compute_sample_salt_and_pepper(sample)
        sample["salt_and_pepper"] = salt_and_pepper


class ComputeSaltAndPepper(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="compute_salt_and_pepper",
            label="Common Issues: compute salt and pepper",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.message(
            "compute salt and pepper", label="compute salt and pepper"
        )
        _execution_mode(ctx, inputs)
        _list_target_views(ctx, inputs)
        return types.Property(inputs)

    def execute(self, ctx):
        view = _get_target_view(ctx, ctx.params["target"])
        compute_dataset_salt_and_pepper(ctx.dataset, view=view)
        ctx.trigger("reload_dataset")


def compute_sample_vignetting(sample):
    # Read the image
    # pylint: disable=no-member
    image = cv2.imread(get_filepath(sample), cv2.IMREAD_GRAYSCALE)

    # Get the image center
    center_y, center_x = np.array(image.shape) / 2

    # Calculate the maximum radius
    max_radius = np.min([center_x, center_y])

    # Create a meshgrid for calculating distances
    y, x = np.ogrid[
        -center_y : image.shape[0] - center_y,
        -center_x : image.shape[1] - center_x,
    ]
    distances = np.sqrt(x**2 + y**2)

    # Calculate the radial intensity profile
    radial_profile = np.array(
        [np.mean(image[distances < r]) for r in range(int(max_radius))]
    )

    # Analyze the profile for a drop-off
    drop_off_percentage = (
        (radial_profile[0] - radial_profile[-1]) / radial_profile[0] * 100
    )
    return drop_off_percentage


def compute_dataset_vignetting(dataset, view=None):
    dataset.add_sample_field("vignetting", fo.FloatField)
    if view is None:
        view = dataset
    for sample in view.iter_samples(autosave=True):
        vignetting = compute_sample_vignetting(sample)
        sample["vignetting"] = vignetting


class ComputeVignetting(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="compute_vignetting",
            label="Common Issues: compute vignetting",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def resolve_input(self, ctx):
        inputs = types.Object()
        inputs.message("compute vignetting", label="compute vignetting")
        _execution_mode(ctx, inputs)
        _list_target_views(ctx, inputs)
        return types.Property(inputs)

    def execute(self, ctx):
        view = _get_target_view(ctx, ctx.params["target"])
        compute_dataset_vignetting(ctx.dataset, view=view)
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
    elif field_name == "salt_and_pepper":
        compute_dataset_salt_and_pepper(dataset)
    elif field_name == "vignetting":
        compute_dataset_vignetting(dataset)
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
    find_issue_images(
        dataset, threshold, "min_exposure", "low_exposure", lt=True
    )


def find_high_exposure_images(dataset, threshold=0.7):
    find_issue_images(
        dataset, threshold, "max_exposure", "high_exposure", lt=False
    )


def find_low_contrast_images(dataset, threshold=50.0):
    find_issue_images(dataset, threshold, "contrast", "low_contrast", lt=True)


def find_high_contrast_images(dataset, threshold=200.0):
    find_issue_images(
        dataset, threshold, "contrast", "high_contrast", lt=False
    )


def find_low_saturation_images(dataset, threshold=40.0):
    find_issue_images(
        dataset, threshold, "saturation", "low_saturation", lt=True
    )


def find_high_saturation_images(dataset, threshold=200.0):
    find_issue_images(
        dataset, threshold, "saturation", "high_saturation", lt=False
    )


def uneven_illumination_images(dataset, threshold=10.0):
    find_issue_images(
        dataset, threshold, "vignetting", "uneven_illumination", lt=False
    )


class FindIssues(foo.Operator):
    @property
    def config(self):
        _config = foo.OperatorConfig(
            name="find_issues",
            label="Common Issues: find issues",
            dynamic=True,
        )
        _config.icon = "/assets/icon.svg"
        return _config

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
                default=50.0,
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
                default=200.0,
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
                default=40.0,
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
                default=200.0,
                label="high saturation threshold",
                view=threshold_view,
            )

        #### UNEVEN ILLUMINATION IMAGES ####
        inputs.bool(
            "uneven_illumination",
            default=True,
            label="Find uneven illumination images in the dataset",
            view=types.CheckboxView(),
        )

        if ctx.params.get("uneven_illumination", False) == True:
            inputs.float(
                "vignetting_threshold",
                default=10.0,
                label="vignetting threshold",
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
                "low_contrast_threshold", 50.0
            )
            find_low_contrast_images(ctx.dataset, low_contrast_threshold)
        if ctx.params.get("high_contrast", False) == True:
            high_contrast_threshold = ctx.params.get(
                "high_contrast_threshold", 200.0
            )
            find_high_contrast_images(ctx.dataset, high_contrast_threshold)
        if ctx.params.get("low_saturation", False) == True:
            low_saturation_threshold = ctx.params.get(
                "low_saturation_threshold", 40.0
            )
            find_low_saturation_images(ctx.dataset, low_saturation_threshold)
        if ctx.params.get("high_saturation", False) == True:
            high_saturation_threshold = ctx.params.get(
                "high_saturation_threshold", 200.0
            )
            find_high_saturation_images(ctx.dataset, high_saturation_threshold)
        if ctx.params.get("uneven_illumination", False) == True:
            vignetting_threshold = ctx.params.get("vignetting_threshold", 10.0)
            uneven_illumination_images(ctx.dataset, vignetting_threshold)

        ctx.trigger("reload_dataset")


def register(plugin):
    plugin.register(ComputeAspectRatio)
    plugin.register(ComputeBlurriness)
    plugin.register(ComputeBrightness)
    plugin.register(ComputeContrast)
    plugin.register(ComputeEntropy)
    plugin.register(ComputeExposure)
    plugin.register(ComputeSaltAndPepper)
    plugin.register(ComputeSaturation)
    plugin.register(ComputeVignetting)
    plugin.register(FindIssues)
