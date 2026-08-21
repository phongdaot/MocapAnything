# Bringing your own rig

Drive **your own rigged character** with an in-the-wild video. Nothing here is
specific to the released datasets — point the pipeline at your FBX and it produces
the same artifacts the shipped `zoo1030` / `obj1k` data is made of, then runs
inference against them.

The worked example below retargets in-the-wild **eagle** footage onto a **parrot**
rig, so it also shows the cross-species case: the input species does not have to
match your rig.

> **No character assets ship with this repo.** The example was built from the
> Truebones Zoo `Parrot`, whose terms of use forbid redistribution. Bring your own
> FBX, or obtain Truebones Zoo from its original source.

---

## What you need

| | |
|---|---|
| A rigged character | one FBX with the mesh + skeleton (the *base*), plus at least one FBX holding a motion |
| Blender | 3.6+; set `BLENDER=/path/to/blender` |
| Python env | the one from [`../../RUN.md`](../../RUN.md) — must have torch, used for the non-Blender stages |
| An input video | any clip of a real animal/person; a matte (no background) works best |

The base FBX is picked automatically: the one with the **shortest filename** in the
folder. Name your files so that holds — e.g. `MyRig.fbx` plus `MyRig-Walk.fbx`.

## Layout

```
Truebone_Z-OO/
└── MyRig/
    ├── MyRig.fbx          # base: mesh + skeleton  (shortest name → detected as base)
    ├── MyRig-Walk.fbx     # one or more motions
    └── tex/               # optional textures
videos/
└── MyRig#clip.mp4         # input footage; the part before '#' must match the rig name
```

The `MyRig#` prefix on the video is how inference finds your reference skeleton, so
it has to match the folder name exactly.

## Run it

```bash
export BLENDER=/path/to/blender
export PYTHON=/path/to/your/python        # the torch env, NOT Blender's

bash examples/custom_rig/run.sh MyRig
```

That is the mesh-free path through [`../../preprocess/run_pipeline.sh`](../../preprocess/run_pipeline.sh)
— stages 1–10 plus 14b, skipping everything that needs `anim_meshes/`. It produces:

```
zoo/
├── fixed_fbx/MyRig/              # 2. no-weight leaf bones removed
├── characters_face_zplus/MyRig/  # 3+4. mesh, skinning, rest.bvh, MyRig_ffs.bvh, front.npy
├── motions_face_zplus/           # 4.  motions rotated to face +Z
├── bvh/MyRig#*/y*.bvh            # 5.  12 yaw views
├── bvh_pose/MyRig#*/y*.npz       # 6.  positions + rot6d + traj
├── species_info_dict.npy         # 7.  topology + joint-name T5 embeddings + static joints
├── cache/…scale_cache.pkl        # 8.  per-species normalization scale
├── video/, image/                # 9+10.
└── npz_train_image_only/         # 14b. DINOv2 reference embeddings
```

Then inference:

```bash
$PYTHON -m inference.video2pose2rot --config examples/custom_rig/inference.yaml
```

Output lands in `outputs/custom_rig/`: a `*_final.mp4` five-panel clip
(input | skeleton | mesh, two camera angles), `*_rot6d_pred.bvh`, `*_pose_pred.npy`,
and per-frame `.obj` meshes.

---

## When the automatic steps get it wrong

Two stages make guesses about your rig, and neither one *fails* when it guesses wrong —
the pipeline runs to completion and the output is just subtly off.

### Facing direction

Stage 4 infers which way your character faces and rotates it to +Z. On rigs with
too much symmetry it gives up:

```
[ERROR] MyRig: Symmetry of size >= 3 found
```

`run.sh` halts there rather than letting the remaining stages build a whole dataset
out of unaligned motions. Supply the facing direction yourself — a unit vector in the
rig's own coordinates pointing out of the character's **front**:

```bash
mkdir -p examples/custom_rig/align_ref/MyRig
$PYTHON -c "import numpy as np; np.save('examples/custom_rig/align_ref/MyRig/front.npy', np.array([1.,0.,0.]))"
ALIGN_REF_DIR=examples/custom_rig/align_ref bash examples/custom_rig/run.sh MyRig
```

`[1,0,0]` means the character faces +X, `[0,0,1]` means it already faces +Z, and so
on. The parrot in the worked example needs `[1,0,0]`.

The case nothing can catch for you is a *confidently wrong* inference: it produces a
`front.npy`, so the run continues, and you only notice because the mesh is rotated in
the output video while the skeleton looks fine. If you see that, set `front.npy`
explicitly and rerun — deleting `$ZOO_ROOT/characters_face_zplus/MyRig/` first, since
stage 4 skips characters it has already aligned.

### Leaf-bone cleanup

Stage 2 deletes leaf bones that carry no skin weights, because that is what the
released datasets were built with — 3ds Max Biped adds a `*Nub` bone at the end of
every chain and they must not become joints. `preprocess/fix_fbx.py` also keeps an
`INVERSE_LIST` of characters whose inferred facing needs flipping; your rig will not
be on it.

If your rig has meaningful weightless leaves, skip the stage:

```bash
STAGES="move_fbx,extract_char,align_faces,rotate_bvh,extract_pose,species_info,scale_cache,render_videos,extract_frames,preprocess_image_only" \
    bash preprocess/run_pipeline.sh
```

### Joint names

`species_info_dict.npy` embeds each joint's *name* with T5, and the model uses those
embeddings to reason about what each joint is. Names are cleaned automatically
(`mixamorig:LeftUpLeg` → `Left Up Leg`), which is exactly how the `obj1k` half of
the training data was built, so descriptive names work well.

Two things to know:

- Trailing digits are stripped, so `Tail_01 … Tail_07` all collapse to `Tail` and
  share one embedding. If ordinals matter for your rig, spell them out.
- The `zoo1030` half of the training data uses curated names (`flyer tail 01`)
  rather than raw ones. To match that convention, hand-write the mapping and pass
  it in:

  ```bash
  $PYTHON preprocess/build_species_info.py --dataset_root zoo \
      --joint_name_map examples/custom_rig/joint_name_map.json
  ```

  The file is `{"MyRig": {"<bone name in the FBX>": "<readable name>", …}}`.

---

## Verified scope

The pipeline was validated end-to-end on a Truebones Zoo rig: the rebuilt
`species_info_dict.npy` matches the released `zoo1030` entry exactly on
`joints_name`, `joints_distance`, `joint_relation`, `rename_clean` and
`static_joints`, with joint-name embeddings at cosine 1.000000.

Not yet validated on Mixamo, Maya HumanIK or hand-built Blender rigs. The leaf-bone
cleanup in particular assumes 3ds Max Biped conventions. If you try one, an issue
report is welcome.
