# Dataset splits

The train/test splits used for the released `video2pose2rot` model.

| File | Used by |
| --- | --- |
| `zoo1030/selected_test_split_release.json` | `configs/train/train_video2pose2rot_multidata.yaml` (`datasets[0].split_json`) |
| `obj1k/select_test_obj1k.json` | `configs/train/train_video2pose2rot_multidata.yaml` (`datasets[1].split_json`) |

Each file maps a sequence name to its test start frame, grouped into `seen` / `rare` /
`unseen` (zoo1030) buckets — the stratification reported in the paper.

These are sequence-name lists only: they contain no motion, mesh, or image data. The
data they index is not in this repository — see the README for where to obtain it.
