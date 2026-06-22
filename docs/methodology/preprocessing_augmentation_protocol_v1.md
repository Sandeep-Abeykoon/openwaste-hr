# OpenWaste-HR Image Preprocessing and Augmentation Protocol v1

## Purpose

This protocol defines how images are preprocessed and augmented during OpenWaste-HR model training and evaluation.

The raw images must remain unchanged. Preprocessing and augmentation are applied during model training and evaluation through the data pipeline.

## Core Rule

Training data uses controlled augmentation.

Validation, test, and unknown evaluation data use deterministic preprocessing only.

This ensures that model training becomes more robust while evaluation remains fair and repeatable.

## Model Input Size

All images are converted to:

- 224 x 224 pixels

This size is suitable for lightweight pretrained backbones such as MobileNetV3, EfficientNet, and ResNet.

## Training Preprocessing

Training images use:

- image decoding as RGB
- random resized crop
- random horizontal flip
- small random rotation
- light colour jitter
- slight affine transformation
- ImageNet normalization

These augmentations help the model handle different lighting, camera angles, backgrounds, and object positions across TrashNet, RealWaste, Garbage Classification V2, and TrashBox.

## Evaluation Preprocessing

Validation, test, and unknown evaluation images use:

- image decoding as RGB
- resize
- center crop or fixed resize
- ImageNet normalization

No random augmentation is used for evaluation.

## Augmentation Limits

Augmentation must remain realistic. The project must not use extreme transformations that change the waste item identity.

Avoid:

- very heavy cropping
- extreme colour shifts
- strong rotation
- excessive blur
- transformations that remove or hide the object

## Active Learning Link

During active learning, uncertain and rejected images are reviewed by a human. If the image is a human-confirmed known class, it can be added to training and will use the same training preprocessing and augmentation pipeline.

True unknown images must remain outside known training.
