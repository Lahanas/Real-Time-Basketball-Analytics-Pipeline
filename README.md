# Real-Time Basketball Analytics Pipeline

Give it a basketball video, get back an annotated video and a stats file.

It tracks every player, splits them into teams by jersey colour, figures out who has the ball, counts passes and interceptions, and maps everyone onto a top-down court to measure speed and distance in real units.

```bash
pip install -e .
courtvision game.mp4
```

## What it does

- **Player tracking**: YOLOv8-pose finds players and their body keypoints; ByteTrack keeps IDs consistent across frames.
- **Teams**: samples each player's jersey colour and clusters them into two teams. No labels or setup needed per game.
- **Ball tracking**: keeps a few ball candidates per frame and follows the one that moves like a ball, filling short gaps.
- **Possession**: looks at whose *hands* the ball is near (from pose keypoints), not just whose box it overlaps. Smoothed so dribbles and brief misses don't flip possession.
- **Passes and interceptions**: detected from changes in possession between teammates or opponents.
- **Court map**: a homography maps players onto a top-down court, which gives the minimap plus speed (km/h) and distance (m) per player. People off the court (bench, crowd) are filtered out.
- **Output**: annotated MP4 with a ball-control bar, pass/steal counters, event pop-ups and a live minimap, plus `stats.json` and `players.csv`.

## Setup

```bash
git clone https://github.com/Lahanas/Real-Time-Basketball-Analytics-Pipeline.git
cd Real-Time-Basketball-Analytics-Pipeline
pip install -e ".[dev]"
```

Default model weights download automatically on first run. Python 3.9+; a GPU is recommended.

## Usage

```bash
courtvision game.mp4                                  # run with defaults
courtvision game.mp4 -c configs/default.yaml -o outputs
courtvision game.mp4 --max-frames 300 --benchmark     # quick test + timing per stage
courtvision game.mp4 --cache cache/game.pkl           # save detections, re-run post-processing instantly
```

Results land in `outputs/`:

| File | What's in it |
|---|---|
| `game_annotated.mp4` | the video with all overlays |
| `game_stats.json` | team possession %, passes, interceptions, every event with a timestamp, per-player stats |
| `game_players.csv` | per-player distance, avg/top speed, possession time, passes |

All settings live in `configs/default.yaml`.

### Mapping the court

Speed, distance, the minimap and off-court filtering need to know where the court is in the image.

- **Fixed camera** (gym, tripod): click a few court landmarks once.
  ```bash
  python tools/calibrate.py game.mp4 -o configs/my_gym.json
  ```
  Then set `court.calibration: configs/my_gym.json` in the config.
- **TV broadcast** (moving camera): use a YOLOv8-pose model trained on court landmarks and set `models.court_keypoint_model`.

Without either, everything else still works.

### Better accuracy on TV footage

The default weights are general-purpose and struggle with small, blurry broadcast balls. For good results, fine-tune a detector on basketball data. [`notebooks/train_benchmark_evaluate.ipynb`](notebooks/train_benchmark_evaluate.ipynb) does it on a free Colab GPU: trains a ball + player model, writes `configs/finetuned.yaml`, benchmarks speed, and evaluates possession.

```bash
courtvision game.mp4 -c configs/finetuned.yaml
```

## Results

| | |
|---|---|
| Detector mAP@0.5 (all / ball) | TBD |
| Speed on NVIDIA T4 | TBD FPS |
| False-positive possessions, pose vs. box-only | TBD |

To measure possession accuracy yourself:

```bash
courtvision clip.mp4 --dump-possession
python evaluation/evaluate_possession.py outputs/clip_possession.json labels.csv
```

where `labels.csv` lists who has the ball (`start_frame,end_frame,player_id`).

## Project layout

```
courtvision/       the pipeline (detection, ball, teams, possession, events, court, stats, render)
configs/           settings
tools/             court calibration tool
evaluation/        possession accuracy script
notebooks/         training + benchmarking on Colab
tests/             unit tests (pytest -q)
```

## Known limitations

- Referees get assigned to whichever team their shirt looks closest to.
- Player IDs can switch after long occlusions, splitting one player's stats in two.
- Speed and distance are only as accurate as the court mapping.

## License

MIT
