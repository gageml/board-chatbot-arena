import argparse
import gdown
import json
import math
import numpy as np
import os
import pandas as pd
import tempfile

def compute_mle_elo(df, SCALE=400, BASE=10, INIT_RATING=1000):
    from sklearn.linear_model import LogisticRegression
    models = pd.concat([df["model_a"], df["model_b"]]).unique()
    models = pd.Series(np.arange(len(models)), index=models)

    # duplicate battles
    df = pd.concat([df, df], ignore_index=True)
    p = len(models.index)
    n = df.shape[0]

    X = np.zeros([n, p])
    X[np.arange(n), models[df["model_a"]]] = +math.log(BASE)
    X[np.arange(n), models[df["model_b"]]] = -math.log(BASE)

    # one A win => two A win
    Y = np.zeros(n)
    Y[df["winner"] == "model_a"] = 1.0

    # one tie => one A win + one B win
    # find tie + tie (both bad) index
    tie_idx = (df["winner"] == "tie") | (df["winner"] == "tie (bothbad)")
    tie_idx[len(tie_idx)//2:] = False
    Y[tie_idx] = 1.0

    lr = LogisticRegression(fit_intercept=False)
    lr.fit(X,Y)

    elo_scores = SCALE * lr.coef_[0] + INIT_RATING

    # calibrate llama-13b to 800
    elo_scores += (800-elo_scores[models["llama-13b"]])
    return pd.Series(elo_scores, index = models.index).sort_values(ascending=False)

def preety_print_model_ratings(ratings):
    df = pd.DataFrame([
        [n, ratings[n]] for n in ratings.keys()
    ], columns=["Model", "Elo rating"]).sort_values("Elo rating", ascending=False).reset_index(drop=True)
    df["Elo rating"] = (df["Elo rating"] + 0.5).astype(int)
    df.index = df.index + 1
    return df

default_repo_dir = os.path.join(tempfile.gettempdir(), "chatbot-arena-leaderboard-old")
file = os.path.join(default_repo_dir, "chatbot_arena_raw_results.json")
summaries_file = os.path.join(default_repo_dir, "chatbot_arena_summary_results.json")

def _sync_repo():
    if not os.path.exists(default_repo_dir):
        url = "https://drive.google.com/file/d/1jjJ8k3L-BzFKSevoGo6yaJ-jCjc2SCK1/view"
        gfile = gdown.download(url, quiet=False, fuzzy=True)
        os.makedirs(default_repo_dir)
        os.rename(gfile, file)
        battles = pd.read_json(file).sort_values(ascending=True, by=["tstamp"])
        elo_mle_ratings = compute_mle_elo(battles)
        with open(summaries_file, "w") as f:
            summary = elo_mle_ratings.to_dict()
            json.dump(summary, f, indent=2, sort_keys=True)

def _model_summary(args):
    name = args.model
    with open(summaries_file) as f:
        summaries = json.load(f)
    elo = summaries[name]
    return {
        "attributes": {
            "model": {
                # "type": "hf:model",
                "value": name,
            },
        },
        "metrics": {
            "elo": {
                "label": "ELO",
                # "type": "hf:open-llm/arc",
                "value": round(elo, 2),
            },
        },
        "run": {
            "label": name,
        },
    }

def _write_summary(summary, args):
    with open(args.summary, "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

def _show_models_and_exit():
    with open(summaries_file) as f:
        summaries = json.load(f)
    for model, result in sorted(summaries.items()):
        print(model, result)
    raise SystemExit(0)

model = None
def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "-m",
        "--model",
        help="Model to import",
        default=model,
    )
    p.add_argument(
        "--list-models",
        help="Show available models and exit",
        action="store_true",
    )
    p.add_argument(
        "--summary",
        help="Summary file to write",
        default="summary.json",
    )
    args = p.parse_args()
    if not args.model and not args.list_models:
        raise SystemExit("Specify either --model or --list-models")
    return args

def main():
    args = _parse_args()
    _sync_repo()
    if args.list_models:
        _show_models_and_exit()
        pass
    summary =_model_summary(args)
    _write_summary(summary, args)

if __name__ == "__main__":
    main()