"""Train / val / test splitting.

A single 70/15/15 split is drawn across ALL indexed images, **grouped by
user_id** so no subject appears in more than one split. HaGRID has the same
person in many images; a naive per-image split would leak a subject's hands from
train into test and inflate the test score. Grouping by user gives an honest
generalization estimate (AD-10).

Determinism comes from the (index, seed) pair, so the split is reproducible
without a manifest; write_manifest() still dumps the assignment to CSV so the
exact split is auditable and freezable alongside a checkpoint.

Note: with a group (user-aware) split we can't hard-stratify by class the way a
plain split can, but HaGRID users hold very few images each in our subset
(median 1), so class balance is preserved in practice -- Splits.summary()
reports the actual per-class counts to confirm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        """Per-class row counts for each split (plus totals)."""
        out = pd.DataFrame(
            {
                "train": self.train["label"].value_counts(),
                "val": self.val["label"].value_counts(),
                "test": self.test["label"].value_counts(),
            }
        ).fillna(0).astype(int)
        out["total"] = out.sum(axis=1)
        out.loc["TOTAL"] = out.sum(axis=0)
        return out


def _group_holdout(
    df: pd.DataFrame, holdout_frac: float, seed: int, group_col: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (rest, holdout) so that no group spans both sides."""
    gss = GroupShuffleSplit(n_splits=1, test_size=holdout_frac, random_state=seed)
    rest_idx, hold_idx = next(gss.split(df, groups=df[group_col].values))
    return df.iloc[rest_idx], df.iloc[hold_idx]


def make_splits(
    index: pd.DataFrame,
    train: float,
    val: float,
    test: float,
    seed: int,
    group_by_user: bool = True,
    stratify: bool = True,
) -> Splits:
    """Three-way split of one combined index into train/val/test.

    group_by_user=True  -> split on user_id (no subject in two splits, AD-10).
    group_by_user=False -> plain per-image split, stratified by class if asked.
    """
    if group_by_user:
        # Stage 1: hold out the test fraction by group.
        trainval, test_df = _group_holdout(index, test, seed, "user_id")
        # Stage 2: carve val out of the remainder (val measured on the whole).
        val_rel = val / (train + val)
        train_df, val_df = _group_holdout(trainval, val_rel, seed, "user_id")
    else:
        strat = index["label"] if stratify else None
        trainval, test_df = train_test_split(
            index, test_size=test, random_state=seed, shuffle=True, stratify=strat
        )
        strat_tv = trainval["label"] if stratify else None
        train_df, val_df = train_test_split(
            trainval, test_size=val / (train + val), random_state=seed,
            shuffle=True, stratify=strat_tv,
        )

    splits = Splits(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
    )
    assert_disjoint(splits, check_users=group_by_user)
    return splits


def assert_disjoint(splits: Splits, check_users: bool = True) -> None:
    """Fail loudly on any UUID (or, if grouped, user) shared across splits."""
    tr, va, te = set(splits.train["uuid"]), set(splits.val["uuid"]), set(splits.test["uuid"])
    assert not (tr & va), f"{len(tr & va)} UUID(s) in BOTH train and val"
    assert not (tr & te), f"{len(tr & te)} UUID(s) leak from train into test"
    assert not (va & te), f"{len(va & te)} UUID(s) leak from val into test"
    if check_users:
        utr, uva, ute = (set(splits.train["user_id"]), set(splits.val["user_id"]),
                         set(splits.test["user_id"]))
        assert not (utr & ute), f"{len(utr & ute)} user(s) leak from train into test"
        assert not (uva & ute), f"{len(uva & ute)} user(s) leak from val into test"
        assert not (utr & uva), f"{len(utr & uva)} user(s) in BOTH train and val"


def write_manifest(splits: Splits, path: str | Path) -> Path:
    """Write a uuid,label,user_id,split CSV so the assignment is auditable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for name, df in ("train", splits.train), ("val", splits.val), ("test", splits.test):
        frames.append(df[["uuid", "label", "user_id"]].assign(split=name))
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path
