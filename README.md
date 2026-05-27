# pi0.5-bimanual

Instructions and openpi config patches for fine-tuning pi0.5 on Trossen `trossen_ai_stationary` bimanual tasks.

Weights live on Hugging Face (private):
- Bimanual flip 5 objects: https://huggingface.co/BruceZhang0912/pi05-bimanual-flip-5-objects

## What's here

- `dev_log.md` — full session log for the bimanual flip fine-tune. Read section 4.B for the eval-desktop runbook.
- `pi05_config_patch.diff` — adds the `pi05_bimanual_flip_5_objects` TrainConfig to upstream openpi (`TrossenRobotics/openpi`, branch `trossen-ai`). Apply with `git apply`.

Future task variants (rotation, etc.) will land here as additional patches + log entries.
