# Does the policy actually use its inputs?

A policy can reach a low training loss and still be blind. These scripts were
written to find out which inputs a trained checkpoint really reads, after
SmolVLA scored 0/50 closed-loop while its loss looked fine. Run them all from
the repo root (they do `sys.path.insert(0, "sim")`) in the `train` env, with
`MUJOCO_GL=egl` for anything that renders:

```bash
MUJOCO_GL=egl pixi run -e train python sim/diagnose/grounding_test.py <ckpt> --seeds 16
```

They are research scripts, not tools: a few have hardcoded checkpoints, seed
lists and episode indices near the top, meant to be edited.

| Script | Question | What it does |
|---|---|---|
| `grounding_test.py` | Does it aim at the cube it was told to pick? | Rolls out, freezes the frame where the gripper starts closing, regresses TCP xy on source-cube xy across seeds. Slope ≈ 1 means the aim follows the cube; ≈ 0 means it reaches to the same spot regardless. Also reports whether the nearest cube at close time is the one the instruction named. |
| `image_ablation.py` | Does it read the cameras? | Holds state and instruction fixed, swaps in another scene's images or grey frames, measures how far the predicted action moves. |
| `instruction_ablation.py` | Does it read the text? | Holds images and state fixed, varies only the instruction: paraphrase, swapped source cube, reversed pair, blank string. |
| `cube_probe.py` | Do the pixels even carry the answer? | Trains a small from-scratch conv net to regress cube xy from the front camera. An information ceiling: if the probe succeeds and the policy fails, the policy has a learning problem, not a perception problem. |
| `gripper_trace.py` | Is the gripper command sane? | Prints min command, first step below the closing threshold, and how long it stays closed. Written to check whether temporal ensembling smears the binary open/closed command (it does not — the effect is lag). |
| `check_img_pipeline.py`, `raw_vs_codec.py` | Do training and eval images match? | Compares dataset frames against freshly rendered ones, and raw renders against video-codec round-trips. |

## Interpreting the numbers

Every measure needs a scale and a positive control, or it means nothing.

- **Ablation shifts** are in degrees of joint angle, against ~12° of action
  standard deviation across the frames tested. Compare models, not absolutes.
- **Grounding slope** needs a model known to work as the reference. Single-pair
  ACT reached 1.03/1.11 (r = 0.99), so slopes near 1 are achievable here and a
  slope of 0.13 is a real failure, not a hard task.

Reference values measured on this dataset (823 episodes, A10):

| | SmolVLA multi-task | SmolVLA single-pair | ACT single-pair |
|---|---|---|---|
| scene-swap shift | 0.78° | 0.95° | **8.05°** |
| instruction-swap shift | 0.29° | 0.54° | n/a (no text input) |
| grounding slope x / y | 0.13 / 0.53 | 0.69 / 0.85 | **1.03 / 1.11** |
| mean aim error | 13.3 cm | 5.6 cm | **2.4 cm** |
| correct cube at close | 8/16 | 15/16 | **16/16** |
| success | 0/50 | 4/50 | **12/50** |

`cube_probe.py` reaches 3.5 cm validation error against a 15.4 cm
predict-the-mean baseline, with depth (2.2 cm) and lateral (2.3 cm) equally
accurate. The cameras carry the cube positions.

## The methodological point

On this task **training loss ordered the models backwards**. Multi-task SmolVLA
reached loss 0.012 and scored 0/50; the single-pair run reached 0.021 and scored
4/50. Same architecture and objective, so the two are directly comparable. Loss
was not just a weak signal, it pointed the wrong way, and several hours went
into runs it had endorsed. These scripts exist because closed-loop behaviour and
input sensitivity are the things worth watching instead.
