# RealWaste Inspection v1 Report

## Summary

| metric | value |
| --- | ---: |
| total_samples | 4752 |
| known_samples | 4434 |
| unknown_samples | 318 |
| missing_path_count | 0 |
| unreadable_path_count | 0 |

## Usage Distribution

| usage | count | proportion |
| --- | --- | --- |
| known_train | 3103 | 0.652988 |
| known_test | 666 | 0.140152 |
| known_val | 665 | 0.139941 |
| unknown_test | 318 | 0.066919 |

## Fine-Label Distribution

| fine_label | count | proportion |
| --- | --- | --- |
| paper_cardboard | 961 | 0.202231 |
| plastic | 921 | 0.193813 |
| organic | 847 | 0.178241 |
| metal | 790 | 0.166246 |
| residual | 495 | 0.104167 |
| glass | 420 | 0.088384 |
| unknown | 318 | 0.066919 |

## Coarse-Label Distribution

| coarse_label | count | proportion |
| --- | --- | --- |
| recyclable | 3092 | 0.650673 |
| organic | 847 | 0.178241 |
| residual | 495 | 0.104167 |
| unknown | 318 | 0.066919 |

## Mapping-Role Distribution

| mapping_role | count | proportion |
| --- | --- | --- |
| known_train_candidate | 4434 | 0.933081 |
| future_class_candidate | 318 | 0.066919 |

## Original Label Distribution

| original_label | count | proportion |
| --- | --- | --- |
| Plastic | 921 | 0.193813 |
| Metal | 790 | 0.166246 |
| Paper | 500 | 0.105219 |
| Miscellaneous Trash | 495 | 0.104167 |
| Cardboard | 461 | 0.097012 |
| Vegetation | 436 | 0.091751 |
| Glass | 420 | 0.088384 |
| Food Organics | 411 | 0.086490 |
| Textile Trash | 318 | 0.066919 |

## Image Dimension Sample Summary

| field | value |
| --- | ---: |
| checked_image_count | 100 |
| readable_image_count | 100 |
| unreadable_image_count | 0 |
| min_width | 524 |
| max_width | 524 |
| min_height | 524 |
| max_height | 524 |

## Key Interpretation

RealWaste contributes expanded known-class data and a public-dataset unknown/future-class split.

The unknown split comes from Textile Trash, which is intentionally kept outside the current known taxonomy.

This supports the OpenWaste-HR open-set design and prepares the project for expanded training.
