import os

import pandas as pd

import wandb

api = wandb.Api()

# Project is specified by <entity/project-name>

entity = ""
project = ""

runs = api.runs(f"{entity}/{project}")

summary_list, config_list, name_list = [], [], []
for run in runs:
    # .summary contains the output keys/values for metrics like accuracy.
    #  We call ._json_dict to omit large files
    summary_list.append(run.summary._json_dict)

    # .config contains the hyperparameters.
    #  We remove special values that start with _.
    config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})

    # .name is the human-readable name of the run.
    name_list.append(run.name)

summary_df = pd.DataFrame.from_dict(pd.json_normalize(summary_list), orient="columns")
config_df = pd.DataFrame.from_dict(pd.json_normalize(config_list), orient="columns")
name_df = pd.DataFrame.from_dict(pd.json_normalize(name_list), orient="columns")
runs_df = pd.concat([name_df, config_df, summary_df], axis=1)
# runs_df = pd.DataFrame(
#     {"summary": summary_list, "config": config_list, "name": name_list}
# )

runs_df.to_csv(f"wandb-export-{project}.csv")
runs_df.to_excel(f"wandb-export-{project}.xlsx")
