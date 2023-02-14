import pandas as pd
import torch
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import os
import pickle
from collections import defaultdict
import matplotlib.pyplot as plt
import logging
from vae_models import *
import sys
def initial_preprocess(name):
    DATA_PATH = "data"
    if name == "adult_income":
        df = pd.read_csv(f"{DATA_PATH}/{name}.csv")
        df.rename(
            columns={
                "fnlwgt": "final_weight",
                "educational-num": "educational_num",
                "marital-status": "marital_status",
                "capital-gain": "capital_gain",
                "capital-loss": "capital_loss",
                "hours-per-week": "hours_per_week",
                "native-country": "native_country",
            },
            inplace=True,
        )
        df.drop(df[df["occupation"] == "?"].index, inplace=True)
        df.drop(["final_weight", "educational_num"], axis=1, inplace=True)
        return df


def save_and_get_dataset_keys(name, trace_func):
    DATA_PATH = "data"
    if name == "adult_income":
        try:
            with open(f"{DATA_PATH}/{name}/map_key.pkl", "rb") as f:
                index_map, dict_keys, gender_idx, num_cols = pickle.load(f)
        except FileNotFoundError as e:
            trace_func(
                f"Did not find the mapping and key files in {DATA_PATH}/{name}/map_key.pkl. Generating them now..."
            )
            df = initial_preprocess(name)
            X, _ = get_dataset(name, return_dataframe=True, df=df)
            num_cols = df.select_dtypes(np.number).columns
            all_cols = [(i, x) for i, x in enumerate(X.columns)]
            relevant_cols = [(i, x) for (i, x) in all_cols if x not in num_cols]
            dict_keys = [x for x in df.columns if x not in num_cols]
            dict_keys.remove("gender")
            dict_keys.remove("income")
            gender_idx = X.columns.get_loc("gender")
            index_map = defaultdict(list)
            for key in dict_keys:
                rel_cols = filter(lambda x: x[1].startswith(key), relevant_cols)
                for col in rel_cols:
                    index_map[key].append(col[0])
            with open(f"{DATA_PATH}/{name}/map_key.pkl", "wb") as f:
                pickle.dump((index_map, dict_keys, gender_idx, num_cols), f)

        return index_map, dict_keys, gender_idx, num_cols


def get_dataset(
    name,
    save_transformed=False,
    random_state=42,
    test_size=0.2,
    return_original_dataframe=False,
    return_dataframe=False,
    return_dataloader=True,
    train_batch_size=64,
    test_batch_size=64,
    df=None,
):
    DATA_PATH = "data"
    if name == "adult_income":
        if df is None:
            df = initial_preprocess(name)
        if return_original_dataframe:
            return df
        categorical_columns = lists = [
            "workclass",
            "education",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "native_country",
        ]
        encoder = ce.OneHotEncoder(cols=categorical_columns, use_cat_names=True)
        df1 = encoder.fit_transform(df)
        df1["gender"] = df1["gender"].apply(lambda x: 1 if x == "Female" else 0)
        df1["income"] = df1["income"].apply(lambda x: 1 if x == ">50K" else 0)
        X = df1.drop("income", axis=1)
        y = df1["income"]
        X_cols = df.select_dtypes(np.number).columns.tolist()
        non_num_cols = [x for x in X.columns if x not in X_cols]
        X_cols.extend(non_num_cols)
        X = X.reindex(columns=X_cols)
        if return_dataframe:
            return X, y
        if save_transformed:
            os.makedirs(f"{DATA_PATH}/{name}/", exist_ok=True)
            X.to_csv(f"{DATA_PATH}/{name}/X_unscaled.csv")
            y.to_csv(f"{DATA_PATH}/{name}/y.csv")

        num_cols = df.select_dtypes(np.number).columns
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        X_train[num_cols] = minmax_scale(X_train[num_cols])
        X_test[num_cols] = minmax_scale(X_test[num_cols])
        yt_train = torch.tensor(y_train.values)
        Xt_train = torch.tensor(X_train.values).float()
        yt_test = torch.tensor(y_test.values)
        Xt_test = torch.tensor(X_test.values).float()
        train = torch.utils.data.TensorDataset(Xt_train, yt_train)
        test = torch.utils.data.TensorDataset(Xt_test, yt_test)
        if not return_dataloader:
            return train, test
        train_loader = torch.utils.data.DataLoader(
            train, batch_size=train_batch_size, shuffle=True
        )
        test_loader = torch.utils.data.DataLoader(
            test, batch_size=test_batch_size, shuffle=False
        )

        return train_loader, test_loader, len(train), len(test), X_train.shape[1], len(num_cols)


def get_split_dataset(
    name, xA, random_state=42, test_size=0.2, train_batch_size=64, test_batch_size=64
):
    if name == "adult_income":
        if xA != "gender":
            raise ValueError(f"For {name}: expected xA = gender but received {xA}")
        df = get_dataset(name, return_original_dataframe=True, return_dataloader=False)
        df_female = df.loc[df["gender"] == "Female"]
        df_male = df.loc[df["gender"] == "Male"]

        (
            train_loader_female,
            test_loader_female,
            len_train_female,
            len_test_female,
            feature_dim_female,
            num_cols_female
        ) = get_dataset(
            name,
            random_state=random_state,
            test_size=test_size,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            df=df_female,
        )
        (
            train_loader_male,
            test_loader_male,
            len_train_male,
            len_test_male,
            feature_dim_male,
            num_cols_male
        ) = get_dataset(
            name,
            random_state=random_state,
            test_size=test_size,
            train_batch_size=train_batch_size,
            test_batch_size=test_batch_size,
            df=df_male,
        )

        return (
            train_loader_female,
            test_loader_female,
            len_train_female,
            len_test_female,
            feature_dim_female,
            num_cols_female,
            train_loader_male,
            test_loader_male,
            len_train_male,
            len_test_male,
            feature_dim_male,
            num_cols_male
        )


def return_argmax(s):
    out = torch.zeros_like(s)
    argmax_idx = torch.argmax(s, axis=1).unsqueeze(1)
    out.scatter_(1, argmax_idx, 1)
    return out


def postprocess(sample, name, trace_func=print):
    if name == "adult_income":
        index_map, dict_keys, gender_idx, _ = save_and_get_dataset_keys(name, trace_func)
        s_ = sample.clone()
        s_[:, gender_idx] = (s_[:, gender_idx] > 0.5).float()
        for key in dict_keys:
            s_[:, index_map[key]] = return_argmax(s_[:, index_map[key]])

        return s_


class EarlyStopping:
    def __init__(
        self, patience=7, verbose=False, delta=0, save_path=None, trace_func=print
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = save_path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        torch.save(model.state_dict(), f"{self.save_path}/best.pt")
        self.val_loss_min = val_loss

def vae_map(vae):
    if vae == 'vae':
        return VanillaVAE, {}
    elif vae == 'rbvae':
        return RelaxedBernoulliVAE, {'tau': 1.0, 'tau_min': 0.1, 'anneal_rate': 3e-5, 'steps': 0, 'hard': False}
    elif vae == 'bvae':
        return BernoulliVAE
    elif vae == 'mgvae':
        return MixedGumbelVae



def plot_loss(epochs, train_loss, val_loss, dataset, run_number, specific_name='', vae='vae'):
    logging.getLogger("matplotlib.font_manager").disabled = True
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, label="train", marker="s")
    plt.plot(epochs, val_loss, label="val", marker="o")
    plt.grid()
    plt.xlabel(r"Epochs $\to$")
    plt.ylabel(r"Loss $\to$")
    plt.legend()
    plt.savefig(f"results/{dataset}/{run_number}_{vae}_{specific_name}_loss_plot.png")
