import pandas as pd


def set_dataset_split(frame: pd.DataFrame, cut_off_period: str) -> pd.DataFrame:
    split_frame = (
        frame
        .assign(period=lambda f: f['creation_date'].apply(lambda x: str(x)[:7]))
        .assign(group=lambda f: f['period'].apply(lambda period: 'training' if period <= cut_off_period else 'test'))
    )

    return split_frame
