# Project Working Rules

## Language

- Use Simplified Chinese for all user-facing chat, including explanations, questions, progress updates, debugging analysis, and final replies.
- Use English for all project-owned files and content: code, identifiers, comments, docstrings, documentation, configuration, experiment records, chart titles, result files, and Git commit messages.
- Preserve the supplied dataset verbatim, including its existing documentation, even when that documentation uses another language.

## Restore Context Before Starting Work

1. Read `docs/PROJECT_CONTEXT.md` and `docs/CURRENT_STATUS.md` before starting a new task.
2. Consult `docs/EXPERIMENT_LOG.md`, existing code, and Git history when relevant.
3. Read `data/bdd100k_subset_10k/README.md` before implementing dataset loading or evaluation. Treat the supplied annotations and `class_mapping.json` as the authoritative dataset contract.
4. Inspect the working tree and existing files before editing. Preserve unrelated user changes and avoid duplicate files.

## Keep Context Current

- Update `docs/CURRENT_STATUS.md` after each important development stage. Record completed work, remaining work, important technical decisions, known issues, and next steps.
- Update `docs/EXPERIMENT_LOG.md` after every completed training experiment with its actual configuration, results, and observations. Record failed or interrupted runs honestly when applicable.
- Update `docs/PROJECT_CONTEXT.md` when a lasting project decision changes.
- Never mark unfinished work as completed. Distinguish plans, verified facts, dataset-provided reports, and measured model results.
- Never invent experiments, metrics, successful checks, or results. Record unavailable values as `Not measured` or `Not evaluated`, with a reason.

## Development Scope and Style

- Complete one development stage at a time, within the user's authorized scope.
- Explain the plan in Simplified Chinese before modifying important code.
- Prefer simple, readable Python and built-in PyTorch and Torchvision functionality. Avoid unnecessary frameworks, abstractions, and infrastructure.
- Run checks appropriate to the change and report their actual outcomes and limitations.
- The initialization stage does not authorize dependency installation, training scripts, pretrained-weight downloads, or model training. Consult the current status and the user's next request before starting another stage.

## Dataset and Evaluation Integrity

- Never delete, move, modify, regenerate, clean, resample, or repartition the supplied dataset without explicit user authorization. Do not edit dataset configuration files for local paths; keep local configuration outside `data/`.
- Preserve the fixed train/validation/test splits, class names, class order, category IDs, image IDs, and subset definitions.
- Use cleaned COCO annotations in `annotations/` for the Faster R-CNN baseline. Do not reconstruct training annotations from `labels_original/`.
- Preserve `bike` and `motor` as supplied. Never reuse the pretrained COCO model's category IDs directly for this dataset.
- Use the training split for fine-tuning and validation for model selection, hyperparameter tuning, and condition analysis.
- Never use the test split or its performance to tune hyperparameters, select checkpoints, or choose improvements. Perform final test evaluation only after the model and hyperparameters are fixed.
- Use standard COCOeval provisionally until the group's shared evaluation script is supplied. Document evaluation settings and any later differences.
- Validation condition subsets overlap and share `images/val/`; do not sum their counts as disjoint groups.
- Validation contains no objects in category `train` (COCO ID 6). Report that class's AP as `Not evaluable`, not zero. Handle any other class absent from an evaluated subset the same way.
- Keep original COCO image IDs and category IDs when exporting predictions. Convert bounding-box formats only in memory when the model/evaluator interface requires it.

## Git and Artifacts

- Keep the entire `data/` and `outputs/` directories ignored by Git. Store future checkpoints, predictions, plots, and run artifacts under `outputs/`.
- Do not force-add ignored data, weights, virtual environments, local secrets, or generated artifacts.
- Check Git status before and after meaningful changes, including tracked files under `data/` and `outputs/`. Ignore rules do not untrack existing files; report any such finding before taking corrective action.
- Obtain explicit user confirmation before every Git commit and before every push. Permission to edit files does not authorize a commit or push.
