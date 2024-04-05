import ast
import argparse
import git
import glob
import json
import pickle
import numpy as np
import os
import pandas as pd
import tempfile

def model_hyperlink(model_name, link):
    return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'

def load_leaderboard_table_csv(filename, add_hyperlink=True):
    lines = open(filename).readlines()
    heads = [v.strip() for v in lines[0].split(",")]
    rows = []
    for i in range(1, len(lines)):
        row = [v.strip() for v in lines[i].split(",")]
        for j in range(len(heads)):
            item = {}
            for h, v in zip(heads, row):
                if h == "Arena Elo rating":
                    if v != "-":
                        v = int(ast.literal_eval(v))
                    else:
                        v = np.nan
                elif h == "MMLU":
                    if v != "-":
                        v = round(ast.literal_eval(v) * 100, 1)
                    else:
                        v = np.nan
                elif h == "MT-bench (win rate %)":
                    if v != "-":
                        v = round(ast.literal_eval(v[:-1]), 1)
                    else:
                        v = np.nan
                elif h == "MT-bench (score)":
                    if v != "-":
                        v = round(ast.literal_eval(v), 2)
                    else:
                        v = np.nan
                item[h] = v
            if add_hyperlink:
                item["Model"] = model_hyperlink(item["Model"], item["Link"])
        rows.append(item)

    return rows

def get_arena_table(arena_df, model_table_df):
    # sort by rating
    arena_df = arena_df.sort_values(by=["rating"], ascending=False)
    values = []
    for i in range(len(arena_df)):
        row = []
        model_key = arena_df.index[i]
        model_name = model_table_df[model_table_df["key"] == model_key]["Model"].values[
            0
        ]

        # rank
        row.append(i + 1)
        # model display name
        row.append(model_name)
        # elo rating
        row.append(round(arena_df.iloc[i]["rating"]))
        upper_diff = round(
            arena_df.iloc[i]["rating_q975"] - arena_df.iloc[i]["rating"]
        )
        lower_diff = round(
            arena_df.iloc[i]["rating"] - arena_df.iloc[i]["rating_q025"]
        )
        row.append(f"+{upper_diff}/-{lower_diff}")
        # num battles
        row.append(round(arena_df.iloc[i]["num_battles"]))
        # Organization
        row.append(
            model_table_df[model_table_df["key"] == model_key]["Organization"].values[0]
        )
        # license
        row.append(
            model_table_df[model_table_df["key"] == model_key]["License"].values[0]
        )

        cutoff_date = model_table_df[model_table_df["key"] == model_key]["Knowledge cutoff date"].values[0]
        if cutoff_date == "-":
            row.append("Unknown")
        else:
            row.append(cutoff_date)
        values.append(row)
    return values

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

def _summary_data():
    elo_result_file = _get_elo_result_file()
    leaderboard_table_file = _get_leaderboard_table_file()

    model_table_data = load_leaderboard_table_csv(leaderboard_table_file, add_hyperlink=False)
    model_table_df = pd.DataFrame(model_table_data)
    with open(elo_result_file, "rb") as fin:
        elo_results = pickle.load(fin)
    arena_df = elo_results["leaderboard_table_df"]
    arena_table_vals = get_arena_table(arena_df, model_table_df)
    data = dict()
    for atv in arena_table_vals:
        i,modl,elo,ci,votes,org,lic,date=atv
        data[modl] = (elo,ci,votes,org,lic,date)
    return data

def _model_summary(args):
    summaries = _summary_data()
    name = args.model
    elo,ci,votes,org,lic,date = summaries[name]
    return {
        "attributes": {
            "model": {
                "value": name,
            },
            "votes": {
                "value": votes,
            },
            "organization": {
                "value": org,
            },
            "license": {
                "value": lic,
            },
            "cutoff_date": {
                "value": date,
            },
        },
        "metrics": {
            "elo": {
                "label": "ELO",
                "value": elo,
            },
            "ci": {
                "label": "95% CI",
                "value": ci,
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
    summaries = _summary_data()
    for modl in sorted(summaries.keys()):
        print(modl)
    raise SystemExit(0)

def _get_elo_result_file():
    elo_result_files = glob.glob(f"{repo_dir}/elo_results_*.pkl")
    elo_result_files.sort(key=lambda x: int(x[43:-4])) # calculated based on hf_repo len
    return elo_result_files[-1]

def _get_leaderboard_table_file():
    leaderboard_table_files = glob.glob(f"{repo_dir}/leaderboard_table_*.csv")
    leaderboard_table_files.sort(key=lambda x: int(x[49:-4])) # calculated based on hf_repo len
    return leaderboard_table_files[-1]

hf_repo = "https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard"
repo_dir = os.path.join(tempfile.gettempdir(), "chatbot-arena-leaderboard")
def _sync_repo():
    if os.path.exists(repo_dir):
        git.Repo(repo_dir).remotes.origin.pull()
    else:
        git.Repo.clone_from(hf_repo, repo_dir)

if __name__ == "__main__":
    args = _parse_args()
    _sync_repo()
    if args.list_models:
        _show_models_and_exit()
    summary = _model_summary(args)
    _write_summary(summary, args)
