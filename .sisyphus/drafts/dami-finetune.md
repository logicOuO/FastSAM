# Draft: dami-finetune

## Requirements (confirmed)
- Goal: Continue from completed COCO training and fine-tune on `dami` for rice/foreign-object segmentation.
- Ignore `shapesData` field in dami JSON.
- `shapes` contains `rectangle` and `enhancePolygon`; planner should decide usage strategy.
- User-approved strategy: Polygon-first (`enhancePolygon`), rectangle fallback.
- Fine-tune initialization: COCO `best.pt` (user-approved).

## Technical Decisions
- Data conversion target format: YOLO segmentation labels.
- Class mapping: dami labels `1/2/3` -> model classes `0/1/2`.
- Preserve extreme aspect-ratio handling considerations for ~1920x64 images.

## Research Findings
- COCO training completed successfully (15 epochs, batch 16) with usable `best.pt`.
- Domain gap validated: direct COCO->dami inference produced no detections on sampled 5 images.
- Existing repo script `converter.py` is COCO-oriented and not directly suitable for dami LabelMe-like JSON.

## Scope Boundaries
- INCLUDE: conversion plan, dataset config plan, fine-tuning plan, QA/verification plan.
- EXCLUDE: re-running COCO training, using `shapesData`, unrelated datasets.

## Open Questions
- Test strategy choice for this work plan:
  - TDD
  - Tests-after
  - No automated tests (agent-executed QA only)
