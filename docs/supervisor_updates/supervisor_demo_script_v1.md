# OpenWaste-HR Supervisor Demo Script v1

## Purpose

This script gives a short walkthrough for demonstrating the OpenWaste-HR prototype to a supervisor.

## Demo Goal

Show that OpenWaste-HR is not only a normal image classifier. It is a hierarchical open-set waste classification prototype with a safe decision policy and human-in-the-loop active learning support.

## Step 1: Explain the Best System

The current best system is:

```text id="c1cm2v"
Pretrained Safe Hierarchical Policy
```

Main result:

| Metric                            |    Value |
| --------------------------------- | -------: |
| Known-test coverage               | 0.864583 |
| Accepted hierarchical reliability | 0.960843 |
| Local unknown manual-review rate  | 0.600000 |
| Local unknown acceptance rate     | 0.400000 |

## Step 2: Start Backend

Open PowerShell in the project root:

```powershell id="43da4w"
cd "D:\Github Repositories\openwaste-hr"
$env:PYTHONPATH="ml/src;."
uvicorn backend.app.main:app --reload --port 8000
```

Expected:

```text id="bgeucc"
Application startup complete.
Uvicorn running on http://127.0.0.1:8000
```

## Step 3: Start Frontend

Open another PowerShell window:

```powershell id="k7ta3c"
cd "D:\Github Repositories\openwaste-hr\frontend"
python -m http.server 5500
```

Open in browser:

```text id="uqrmxc"
http://127.0.0.1:5500
```

## Step 4: Test Image

Use:

```text id="7ns0xc"
ml/data/local_unknown/images/local_000001.jpg
```

This image is a rubber slipper / flip-flop and is outside the current known taxonomy.

## Step 5: Expected Demo Output

The current system returns:

| Field                  | Value           |
| ---------------------- | --------------- |
| predicted fine label   | paper_cardboard |
| max softmax confidence | 0.993654        |
| decision type          | coarse_label    |
| final label            | recyclable      |
| final confidence       | 0.999999        |

## Step 6: Explain the Research Meaning

Although the model is highly confident, the object is actually outside the current taxonomy.

This demonstrates why the project needs:

* local unknown evaluation
* hierarchical fallback decisions
* manual-review routing
* active learning for reviewed local samples

## Step 7: Show Active Learning Decision

The reviewed local sample is handled as:

| Field                   | Value                                   |
| ----------------------- | --------------------------------------- |
| human observation       | rubber slipper / flip-flop              |
| taxonomy status         | outside_current_known_taxonomy          |
| recommended action      | keep_as_unknown_test                    |
| active learning v2 role | unknown_test_and_future_class_candidate |

This prevents the image from being incorrectly used as a known training sample.

## Demo Closing Line

OpenWaste-HR improves waste classification by combining pretrained image recognition with hierarchical open-set decision-making, safe reject/manual-review behaviour, local unknown evaluation, and human-in-the-loop active learning.
