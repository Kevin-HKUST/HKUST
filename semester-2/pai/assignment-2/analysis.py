import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Enable CoW optimizations
pd.options.mode.copy_on_write = True


def get_random_rows(df: pd.DataFrame, num_rows: int = 10, random_state: int = 5101) -> pd.DataFrame:
    """Samples `num_rows` of data from the given DataFrame."""
    return df.sample(n=num_rows, random_state=random_state)


def count_feature_types(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Counts the number of features of each type in the dataset.
    """
    numerical = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical = df.select_dtypes(exclude=[np.number]).columns.tolist()
    return numerical, categorical


def convert_to_categorical(df: pd.DataFrame, l: str | list[str]) -> pd.DataFrame:
    """
    Converts the list of input features to categorical features.
    """
    if isinstance(l, str):
        l = [l]
    for col in l:
        df[col] = df[col].astype("category")
    return df


def convert_rain(df: pd.DataFrame) -> pd.DataFrame:
    """Converts the `RainToday` and `RainTomorrow` features to a `bool` Series."""
    for col in ["RainToday", "RainTomorrow"]:
        df[col] = df[col].map({"Yes": True, "No": False}).fillna(False).astype(bool)
    return df


def convert_to_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Converts the `Date` feature into a Series of `datetime` objects."""
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def split_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Derives and adds the `Day`, `Month` and `Year` Series to the DataFrame."""
    df["Day"] = df["Date"].dt.day.astype("int32")
    df["Month"] = df["Date"].dt.month.astype("int32")
    df["Year"] = df["Date"].dt.year.astype("int32")
    return df


def convert_month(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the `Month` Series to an ordered categorical variable with human-readable names.
    """
    month_names = [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ]
    month_map = {i + 1: name for i, name in enumerate(month_names)}
    df["Month"] = df["Month"].map(month_map)
    df["Month"] = pd.Categorical(df["Month"], categories=month_names, ordered=True)
    return df


def month_to_season(month: int) -> str:
    """Converts an integer representing a month into a season string (Southern hemisphere)."""
    if month in [12, 1, 2]:
        return "Summer"
    elif month in [3, 4, 5]:
        return "Autumn"
    elif month in [6, 7, 8]:
        return "Winter"
    elif month in [9, 10, 11]:
        return "Spring"


def add_season_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a `Season` feature to the dataset based on the month."""
    df["Season"] = df["Date"].dt.month.map(month_to_season)
    df["Season"] = pd.Categorical(
        df["Season"],
        categories=["Summer", "Autumn", "Winter", "Spring"],
        ordered=False
    )
    return df


def compute_monthly_rain_days(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the number of days with and without rain per month.
    """
    grouped = df.groupby("Month")["RainTomorrow"]
    has_rain = grouped.sum().astype(int)
    no_rain = grouped.count() - has_rain
    result = pd.DataFrame({
        "HasRainTomorrow": has_rain,
        "NoRainTomorrow": no_rain
    })
    return result


def plot_rain_distrib(df: pd.DataFrame) -> tuple:
    """Plots a stacked bar graph using the `HasRainTomorrow` and `NoRainTomorrow` attributes."""
    fig, ax = plt.subplots(figsize=(7, 7))
    months = df.index.tolist()
    has_rain = df["HasRainTomorrow"].values
    no_rain = df["NoRainTomorrow"].values

    ax.bar(months, has_rain, label="HasRainTomorrow")
    ax.bar(months, no_rain, bottom=has_rain, label="NoRainTomorrow")
    ax.legend()
    ax.set_xticklabels(months, rotation=45, ha="right")
    return fig, ax


def compute_seasonal_rainfall(df: pd.DataFrame) -> pd.Series:
    """Computes the total rainfall per season."""
    return df.groupby("Season")["Rainfall"].sum()


def find_outlier_features(df: pd.DataFrame, exclude_list: list[str]) -> list[str]:
    """Finds the features with outliers using the 1.5 IQR rule."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in exclude_list]

    outlier_features = []
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        if ((df[col] < lower) | (df[col] > upper)).any():
            outlier_features.append(col)
    return outlier_features


def cap_outliers(df: pd.DataFrame, outliers: list[str]) -> pd.DataFrame:
    """Caps all outliers to values within [-1.5*IQR + Q1, Q3 + 1.5*IQR]."""
    for col in outliers:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def impute_data(df: pd.DataFrame) -> pd.DataFrame:
    """Imputes all missing values: mean for numerical, mode for categorical."""
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype.name == "category":
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            elif pd.api.types.is_bool_dtype(df[col]):
                df[col] = df[col].fillna(False)
            else:
                mode_val = df[col].mode()[0]
                df[col] = df[col].fillna(mode_val)
    return df


def plot_corr_heatmap(df: pd.DataFrame) -> tuple:
    """Plots a heatmap between the numerical features, excluding `RISK_MM` and `Date`."""
    exclude = ["RISK_MM", "Date"]
    numeric_df = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in exclude if c in df.select_dtypes(include=[np.number]).columns],
        errors="ignore"
    )
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(corr, cmap="coolwarm", aspect="auto")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    fig.colorbar(im, ax=ax)
    return fig, ax


def select_k_best(df: pd.DataFrame, target: str, k: int) -> list[str]:
    """Selects the k best features most correlated with the target."""
    exclude = ["RISK_MM", "Date"]
    numeric_df = df.select_dtypes(include=[np.number]).drop(
        columns=[c for c in exclude if c in df.select_dtypes(include=[np.number]).columns],
        errors="ignore"
    )
    corr = numeric_df.corr()[target].drop(target).abs().sort_values(ascending=False)
    return corr.head(k).index.tolist()


def find_right_skewed(df: pd.DataFrame) -> pd.Series:
    """Finds all non-boolean numerical features with skewness > 0.5."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Exclude boolean columns
    bool_cols = df.select_dtypes(include=[bool]).columns.tolist()
    cols = [c for c in numeric_cols if c not in bool_cols]

    skew_vals = df[cols].skew()
    return skew_vals[skew_vals > 0.5]


def apply_log_transform(df: pd.DataFrame) -> pd.DataFrame:
    """Applies log(x+1) transformation to all right-skewed features."""
    skewed = find_right_skewed(df)
    for col in skewed.index:
        df[col] = np.log1p(df[col])
    return df


if __name__ == "__main__":
    print("This is a library. Please import it instead of running it.")