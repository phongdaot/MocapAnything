import argparse
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm


def collect_fbx_files(input_root):
    jobs = []
    for char_name in sorted(os.listdir(input_root)):
        char_dir = os.path.join(input_root, char_name)
        if not os.path.isdir(char_dir):
            continue
        for name in sorted(os.listdir(char_dir)):
            if name.lower().endswith(".fbx"):
                jobs.append((char_name, os.path.join(char_dir, name)))
    return jobs


def mode_or_empty(values):
    if not values:
        return ()
    return Counter(values).most_common(1)[0][0]


def write_tsv(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")


def run_one(args_tuple):
    (
        char_name,
        inpath,
        input_root,
        output_root,
        blender,
        script,
        skip_existing,
        analysis_only,
        prefer,
        expected_clean_signature,
        pass_name,
    ) = args_tuple
    rel = os.path.relpath(inpath, input_root)
    outpath = os.path.join(output_root, rel)
    meta_dir = os.path.join(output_root, f"_fix_meta_{pass_name}", char_name)
    log_dir = os.path.join(output_root, f"_fix_logs_{pass_name}", char_name)
    stem = os.path.splitext(os.path.basename(inpath))[0]
    meta_path = os.path.join(meta_dir, stem + ".json")
    log_path = os.path.join(log_dir, stem + ".log")

    if (
        skip_existing
        and not analysis_only
        and os.path.exists(outpath)
        and os.path.exists(meta_path)
    ):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            if meta.get("success"):
                meta["status"] = "skipped"
                meta["species"] = char_name
                meta["relpath"] = rel
                return meta
        except Exception:
            pass

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        blender,
        "--background",
        "--python",
        script,
        "--",
        "--in",
        inpath,
        "--out",
        outpath,
        "--log",
        log_path,
        "--meta",
        meta_path,
    ]
    if analysis_only:
        cmd.append("--analysis-only")
    if prefer:
        cmd.extend(["--prefer", ",".join(prefer)])
    if expected_clean_signature:
        cmd.extend(["--expect-clean-signature", expected_clean_signature])

    try:
        subprocess.run(cmd, check=True)
        with open(meta_path, "r") as f:
            meta = json.load(f)
        meta["species"] = char_name
        meta["relpath"] = rel
        if not analysis_only and not os.path.exists(outpath):
            raise FileNotFoundError(f"Fixer did not create output: {outpath}")
        meta["status"] = "done"
        return meta
    except Exception as exc:
        meta = {
            "input": inpath,
            "output": outpath,
            "success": False,
            "analysis_only": analysis_only,
            "locator": [],
            "cleanup": {"shortened": 0, "deleted": 0},
            "error": str(exc),
            "status": "failed",
            "species": char_name,
            "relpath": rel,
        }
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta.update(json.load(f))
                meta["status"] = "failed"
                meta["species"] = char_name
                meta["relpath"] = rel
            except Exception:
                pass
        if os.path.exists(outpath):
            os.remove(outpath)
        return meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--blender", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--single_pass", action="store_true",
                        help="run a single export pass without species-level preference voting")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    script = os.path.join(repo_root, "preprocess", "fix_fbx.py")
    jobs = collect_fbx_files(args.input_root)
    print(f"==> Fixing {len(jobs)} FBX files with {args.workers} workers")

    def run_tasks(task_args, desc):
        results = []
        if args.workers <= 1:
            for task in tqdm(task_args, desc=desc):
                results.append(run_one(task))
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(run_one, task) for task in task_args]
                for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
                    results.append(fut.result())
        done = sum(1 for item in results if item.get("success"))
        failed = len(results) - done
        print(f"==> {desc}: success={done}, failed={failed}")
        return results

    if args.single_pass:
        pass2_tasks = [
            (
                char_name, inpath, args.input_root, args.output_root, args.blender,
                script, args.skip_existing, False, [], "", "pass2",
            )
            for char_name, inpath in jobs
        ]
        pass2_results = run_tasks(pass2_tasks, "fix_fbx_export")
        failed_rows = [
            (os.path.relpath(item["input"], args.input_root), item.get("error", ""))
            for item in pass2_results if not item.get("success")
        ]
        write_tsv(os.path.join(args.output_root, "_fix_failed.tsv"), failed_rows)
        return

    pass1_tasks = [
        (
            char_name, inpath, args.input_root, args.output_root, args.blender,
            script, False, True, [], "", "pass1",
        )
        for char_name, inpath in jobs
    ]
    pass1_results = run_tasks(pass1_tasks, "fix_fbx_analyze")

    by_species = defaultdict(list)
    for result in pass1_results:
        if result.get("success"):
            char_name = result.get("species", "")
            locator = tuple(result.get("locator") or [])
            cleanup = result.get("cleanup") or {}
            signature = (int(cleanup.get("shortened", 0)), int(cleanup.get("deleted", 0)))
            by_species[char_name].append((locator, signature))

    species_pref = {}
    preference_rows = [("species", "prefer", "cleanup_signature", "valid_count", "invalid_count")]
    for char_name in sorted({name for name, _ in jobs}):
        records = by_species.get(char_name, [])
        locator_mode = mode_or_empty([record[0] for record in records])
        cleanup_mode = mode_or_empty([record[1] for record in records])
        species_pref[char_name] = (locator_mode, cleanup_mode)
        invalid_count = sum(1 for name, _ in jobs if name == char_name) - len(records)
        preference_rows.append(
            (
                char_name,
                ",".join(locator_mode),
                ",".join(str(x) for x in cleanup_mode),
                len(records),
                invalid_count,
            )
        )
    write_tsv(os.path.join(args.output_root, "_fix_preferences.txt"), preference_rows)

    pass2_tasks = []
    for char_name, inpath in jobs:
        locator_mode, cleanup_mode = species_pref.get(char_name, ((), ()))
        if not cleanup_mode:
            continue
        pass2_tasks.append(
            (
                char_name,
                inpath,
                args.input_root,
                args.output_root,
                args.blender,
                script,
                args.skip_existing,
                False,
                list(locator_mode),
                f"{cleanup_mode[0]},{cleanup_mode[1]}",
                "pass2",
            )
        )
    pass2_results = run_tasks(pass2_tasks, "fix_fbx_export")

    failed_rows = [("input", "error")]
    for result in pass1_results + pass2_results:
        if not result.get("success"):
            failed_rows.append(
                (
                    os.path.relpath(result.get("input", ""), args.input_root),
                    result.get("error", ""),
                )
            )
            outpath = result.get("output", "")
            if outpath and os.path.exists(outpath):
                os.remove(outpath)
    write_tsv(os.path.join(args.output_root, "_fix_failed.tsv"), failed_rows)


if __name__ == "__main__":
    main()
