# Dataset Splits

This folder will store split files and protocols.

Do not store large image datasets here.

Expected future files:

| File | Purpose |
|---|---|
| known_train.csv | Training images from known classes |
| known_val.csv | Validation images from known classes |
| known_test.csv | Testing images from known classes |
| unknown_test.csv | Unknown/rejected test images |
| local_unknown.csv | Local phone-captured unknown or difficult images |
| active_learning_batch_01.csv | Human-corrected samples for adaptation experiment |

## Important

The first baseline must use only known classes for training.

Unknown samples are used for open-set evaluation, rejection testing, and active learning.