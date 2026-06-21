# Pretrained Training Smoke Test v1

## Purpose

This stage prepares and runs a small smoke test for the pretrained transfer-learning model.

The aim is not to produce the final pretrained model result yet. The aim is to check whether the pretrained training configuration can run safely before full training.

## Model Being Tested

The current original model is:

```text id="1a641k"
Baseline A = scratch-trained TrashNet-style baseline
```

The next model is:

```text id="wvyxt7"
Baseline B = pretrained transfer-learning baseline
```

This smoke test uses:

```text id="d5z5y4"
ml/configs/train_pretrained_trashnet_smoke.yaml
```

## Key Config Difference

The smoke-test config keeps the pretrained setting enabled:

```yaml id="vj2cw3"
pretrained: true
```

The output name is separated from the real pretrained model:

```text id="wrlcrp"
pretrained_trashnet_smoke_v1
```

This avoids mixing smoke-test files with final training outputs.

## Why a Smoke Test Is Needed

Pretrained training may need to download or load pretrained model weights. It may also reveal problems with model creation, checkpoint paths, output directories, or GPU memory.

A smoke test helps detect these problems before running the full pretrained training experiment.

## What the Smoke Test Checks

| Check                        | Purpose                                            |
| ---------------------------- | -------------------------------------------------- |
| pretrained config loads      | confirm YAML config is valid                       |
| pretrained model initialises | confirm pretrained model can be created            |
| training loop starts         | confirm the script can run with pretrained weights |
| output directory is separate | avoid overwriting existing baseline outputs        |
| checkpoint saving works      | confirm training artefacts can be written          |

## Expected Result

The expected result is a short training run that completes successfully and saves outputs under a smoke-test output name.

This is not the final pretrained comparison result. The final comparison will only be made after full pretrained training and evaluation.

## Possible Issues

| Issue                            | Meaning                                                               |
| -------------------------------- | --------------------------------------------------------------------- |
| pretrained weight download error | pretrained weights may not be cached or internet access may be needed |
| CUDA memory error                | batch size may be too large for the pretrained model                  |
| config key mismatch              | smoke config may need manual adjustment                               |
| output path conflict             | output names must remain separate from the original baseline          |

## Research Meaning

This stage prepares the next model-improvement phase. It checks whether Baseline B can be trained using pretrained transfer learning before the full training experiment is started.
