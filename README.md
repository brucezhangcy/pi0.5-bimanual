# pi0.5-bimanual

Instructions and openpi config patches for fine-tuning pi0.5 on Trossen `trossen_ai_stationary` bimanual tasks.

Weights live on Hugging Face (private):
- Bimanual flip 5 objects: https://huggingface.co/BruceZhang0912/pi05-bimanual-flip-5-objects

## What's here

- `dev_log.md` — full session log. §4.B = eval-desktop runbook; §8 = real-eval session on `coldbrew` (first real run did NOT succeed; see §8.6).
- `pi05_config_patch.diff` — adds the `pi05_bimanual_flip_5_objects` TrainConfig to upstream openpi (`TrossenRobotics/openpi`, branch `trossen-ai`). Apply with `git apply`.
- `trossen_ai_client_patch.diff` — eval-client edits for `openpi/examples/trossen_ai/` (camera serials, `trossen-arm==1.10.0` pin to match firmware, Ctrl+C returns arms to rest). Apply with `git apply` inside `openpi/`.
- `eval_tools/` — safety ladder + scripts (sim playback/limit check, replay overlay, demo trajectory). See `eval_tools/README.md`.

Future task variants (rotation, etc.) will land here as additional patches + log entries.
