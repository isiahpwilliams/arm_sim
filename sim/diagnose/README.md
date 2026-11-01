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
| `cond_ablation.py` | Does a `--task-cond` policy read the (src, dst) one-hot? | The mirror of `instruction_ablation.py` for policies with no language input: holds images and proprioception fixed, varies only the 6 conditioning dims. `--first N` restricts to each episode's opening frames, where the label is load-bearing; sampling whole episodes understates it, because by mid-trajectory `qpos` already encodes which cube the arm is heading to. |
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

| | SmolVLA multi-task | SmolVLA single-pair | ACT single-pair | ACT task-conditioned |
|---|---|---|---|---|
| episodes | 198, 6 pairs | 153, 1 pair | 153, 1 pair | 823, 6 pairs |
| scene-swap shift | 0.78° | 0.95° | **8.05°** | **8.71°** |
| task-swap shift | 0.29° (text) | 0.54° (text) | n/a (no conditioning) | 0.21° (one-hot) |
| grounding slope x / y | 0.13 / 0.53 | 0.69 / 0.85 | **1.03 / 1.11** | 0.42 / 0.42 |
| mean aim error | 13.3 cm | 5.6 cm | **2.4 cm** | — |
| correct cube at close | 8/16 | 15/16 | **16/16** | 6/16 (chance) |
| success | 0/50 | 4/50 | **12/50** | 1/50 |

The last column is the useful contrast: strong vision, dead conditioning. Any
measure of "does it use its inputs" has to name *which* input, because these two
failures are independent and a policy can have one without the other.

`cube_probe.py` reaches 3.5 cm validation error against a 15.4 cm
predict-the-mean baseline, with depth (2.2 cm) and lateral (2.3 cm) equally
accurate. The cameras carry the cube positions.

## How many seeds an eval needs

`eval_policy.py` is deterministic: the same checkpoint on the same seeds returns
the same count (040000 gave 10/50 twice). So the spread between evals is pure
sampling noise, and it is larger than it looks. At n=50 and p around 0.25 the
binomial standard error is **6 percentage points**, so a 50-seed result carries a
95% interval about 25 percentage points wide — 12/50 and 8/50 are the same
measurement. Scoring act_gb's checkpoints at 50 seeds produced 16%, 32%, 20% for
20k/30k/40k, a "curve" that is mostly noise.

Use **n=200** (standard error ~3 points) for any comparison you intend to act on,
and treat n=50 as a smoke test. Two 200-seed proportions 8 points apart are still
only about 1.8 sigma, so a real improvement needs to be large or the sample
bigger. A 200-seed eval is roughly 30 minutes.

Measured properly, single-pair ACT peaks at **30% (60/200) at 30k steps** and
falls to **22% (44/200) by 40k**, while training loss keeps falling — the
checkpoint to beat is 30%, not the 24% an early 50-seed eval suggested.

## The methodological point

On this task **training loss ordered the models backwards**. Multi-task SmolVLA
reached loss 0.012 and scored 0/50; the single-pair run reached 0.021 and scored
4/50. Same architecture and objective, so the two are directly comparable. Loss
was not just a weak signal, it pointed the wrong way, and several hours went
into runs it had endorsed. These scripts exist because closed-loop behaviour and
input sensitivity are the things worth watching instead.
