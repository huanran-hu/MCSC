import json
import math
import argparse
from collections import Counter


def load_json(path):
    """Load a JSON file and return the parsed object."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_video_ids(script):
    """
    Extract video IDs from shots that reference a material.
    Returns a list of video ID strings (without .mp4 suffix).
    Shots without material_usage are skipped.
    """
    video_ids = []
    for shot in script:
        if "material_usage" in shot and "video_id" in shot["material_usage"]:
            vid = shot["material_usage"]["video_id"]
            video_ids.append(vid.replace(".mp4", "") if vid.endswith(".mp4") else vid)
    return video_ids


def calc_err(video_ids, distractor_set):
    """
    Distractor error rate (Err).
    Fraction of material-using shots that incorrectly reference a distractor clip.
    """
    if not video_ids:
        return 0.0
    err_count = sum(1 for v in video_ids if v in distractor_set)
    return err_count / len(video_ids)


def calc_rep(video_ids):
    """
    Material repetition rate (Rep).
    For each clip used more than once, count the excess occurrences,
    then divide by the total number of material-using shots.
    e.g. clip A used 3 times -> contributes (3-1)=2 to the numerator.
    """
    if not video_ids:
        return 0.0
    counter = Counter(video_ids)
    excess = sum(c - 1 for c in counter.values() if c > 1)
    return excess / len(video_ids)


def calc_t(script, gt_duration):
    """
    Duration deviation (T).
    Absolute relative error between the predicted total duration
    (sum of all shot durations) and the ground-truth duration.
    """
    pred_duration = round(sum(float(shot["duration"]) for shot in script), 0)
    return abs(pred_duration - gt_duration) / gt_duration


def evaluate(script_path, metadata_path, sample_id):
    """
    Compute three rule-based metrics for a generated script:
      - Err: distractor error rate
      - Rep: material repetition rate
      - T:   duration deviation
    """
    script = load_json(script_path)
    metadata = load_json(metadata_path)

    if sample_id not in metadata:
        raise KeyError(f"sample_id '{sample_id}' not found in metadata.")

    meta = metadata[sample_id]
    distractor_set = set(str(d) for d in meta["distractor"])
    gt_duration = meta["duration"]

    video_ids = extract_video_ids(script)

    return {
        "Err": round(calc_err(video_ids, distractor_set), 4),
        "Rep": round(calc_rep(video_ids), 4),
        "T": round(calc_t(script, gt_duration), 4),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Rule-based evaluation for generated video scripts."
    )
    parser.add_argument(
        "--script", type=str, required=True,
        help="Path to the generated script JSON file."
    )
    parser.add_argument(
        "--metadata", type=str, required=True,
        help="Path to metadata.json (contains distractor list and target duration)."
    )
    parser.add_argument(
        "--sample_id", type=str, required=True,
        help="Sample ID to look up in metadata."
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to save the evaluation result JSON."
    )
    args = parser.parse_args()

    result = evaluate(args.script, args.metadata, args.sample_id)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
