# CODSOFT DATA SCIENCE INTERNSHIP
# TASK 2 : MOVIE RATING PREDICTION WITH PYTHON

import warnings
warnings.filterwarnings("ignore")

import re
import pickle
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold
)

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

plt.style.use("ggplot")
np.random.seed(42)


# LOAD DATASET
df = pd.read_csv("IMDb Movies India.csv", encoding="latin-1")

print("="*60)
print("DATASET LOADED SUCCESSFULLY")
print("="*60)

print("Rows :", df.shape[0])
print("Columns :", df.shape[1])

print("\nDataset Info")
print(df.info())

print("\nMissing Values")
print(df.isnull().sum())


# DATA CLEANING
df = df[df["Rating"].notna()].copy()

df["Year"] = (
    df["Year"]
    .astype(str)
    .str.extract("(\d{4})")
)

df["Year"] = pd.to_numeric(
    df["Year"],
    errors="coerce"
)


df["Duration"] = (
    df["Duration"]
    .astype(str)
    .str.extract("(\d+)")
)

df["Duration"] = pd.to_numeric(
    df["Duration"],
    errors="coerce"
)



df["Votes"] = (
    df["Votes"]
    .astype(str)
    .str.replace(",", "")
)

df["Votes"] = pd.to_numeric(
    df["Votes"],
    errors="coerce"
)



df["Year"].fillna(df["Year"].median(), inplace=True)
df["Duration"].fillna(df["Duration"].median(), inplace=True)
df["Votes"].fillna(df["Votes"].median(), inplace=True)


# FEATURE ENGINEERING


# Log Votes

df["LogVotes"] = np.log1p(df["Votes"])

# Decade

df["Decade"] = (
    df["Year"] // 10 * 10
)

# Genre Count

df["Genre"] = df["Genre"].fillna("Unknown")

df["GenreCount"] = df["Genre"].apply(
    lambda x: len(str(x).split(","))
)

# Primary Genre

df["PrimaryGenre"] = (
    df["Genre"]
    .str.split(",")
    .str[0]
    .str.strip()
)

# Frequency Encoding Function

def frequency_encode(series):
    freq = series.value_counts(normalize=True)
    return series.map(freq)

# Director Encoding

df["Director"] = df["Director"].fillna("Unknown")

df["DirectorFreq"] = frequency_encode(
    df["Director"]
)

# Actor Encoding

for col in ["Actor 1","Actor 2","Actor 3"]:

    df[col] = df[col].fillna("Unknown")

    df[col+"_Freq"] = frequency_encode(
        df[col]
    )

# Cast Popularity

df["CastPopularity"] = (
    df["Actor 1_Freq"]*0.5 +
    df["Actor 2_Freq"]*0.3 +
    df["Actor 3_Freq"]*0.2
)

# Top Genres

top_genres = (
    df["PrimaryGenre"]
    .value_counts()
    .head(10)
    .index
)

for genre in top_genres:

    df[f"Genre_{genre}"] = (
        df["PrimaryGenre"] == genre
    ).astype(int)


# EDA
plt.figure(figsize=(7,5))

sns.histplot(
    df["Rating"],
    bins=30,
    kde=True
)

plt.title("Movie Rating Distribution")
plt.show()

plt.figure(figsize=(7,5))

sns.scatterplot(
    x="LogVotes",
    y="Rating",
    data=df
)

plt.title("Votes vs Rating")
plt.show()

plt.figure(figsize=(8,5))

top_directors = (
    df["Director"]
    .value_counts()
    .head(10)
)

sns.barplot(
    x=top_directors.values,
    y=top_directors.index
)

plt.title("Top Directors")
plt.show()

director_stats = (

    df.groupby("Director")
    .agg({

        "Rating":"mean",
        "Votes":"mean",
        "Name":"count"

    })

)

director_stats.columns = [

    "AvgRating",
    "AvgVotes",
    "MovieCount"

]

top_director_stats = (
    director_stats[
        director_stats["MovieCount"] >= 5
    ]
    .sort_values(
        "AvgRating",
        ascending=False
    )
)

print("\nTop Directors (Minimum 5 Movies)")
print(top_director_stats.head(10))
genre_rating = (

    df.groupby(
        "PrimaryGenre"
    )["Rating"]

    .mean()

    .sort_values(
        ascending=False
    )

)

genre_rating.head(15).plot(
    kind="bar"
)

plt.title(
    "Average Rating by Genre"
)

plt.show()
# Correlation Heatmap

corr_cols = [
    "Rating",
    "Year",
    "Duration",
    "LogVotes",
    "GenreCount",
    "CastPopularity"
]

plt.figure(figsize=(8,6))

sns.heatmap(
    df[corr_cols].corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")
plt.show()


year_rating = (
    df.groupby("Year")["Rating"]
    .mean()
    .sort_values(ascending=False)
)

print("\nTop 10 Years by Average Rating")
print(year_rating.head(10))

plt.figure(figsize=(12,5))
year_rating.head(10).plot(kind="bar")
plt.title("Top Years by Average Rating")
plt.ylabel("Average Rating")
plt.xlabel("Year")
plt.show()

top_movies = (
    df[["Name","Rating"]]
    .sort_values("Rating", ascending=False)
    .head(10)
)

print("\nTop 10 Movies Overall")
print(top_movies)


top_per_year = (
    df.sort_values(
        ["Year","Rating"],
        ascending=[True,False]
    )
    .groupby("Year")
    .head(10)
)

print(top_per_year[
    ["Year","Name","Rating"]
].head(50))

best_movie_per_year = (
    df.loc[
        df.groupby("Year")["Rating"]
        .idxmax()
    ]
)

print(
    best_movie_per_year[
        ["Year","Name","Rating"]
    ]
)

plt.figure(figsize=(12,6))

sns.barplot(
    data=best_movie_per_year.sort_values("Rating", ascending=False).head(15),
    x="Rating",
    y="Name"
)

plt.title("Top Best Movies Across Different Years")
plt.show()

popular_movies = df[df["Rating"] >= 7]

popular_year = (
    popular_movies
    .groupby("Year")
    .size()
)

plt.figure(figsize=(12,5))

popular_year.plot()

plt.title(
    "Popular Movies Released Per Year"
)

plt.ylabel("Count")

plt.show()

top_votes = (
    df[
        ["Name","Votes","Rating"]
    ]
    .sort_values(
        "Votes",
        ascending=False
    )
    .head(10)
)

print("\nMost Voted Movies")
print(top_votes)

actors = pd.concat([

    df["Actor 1"],
    df["Actor 2"],
    df["Actor 3"]

])

top_actors = actors.value_counts().head(15)

plt.figure(figsize=(10,6))

sns.barplot(

    x=top_actors.values,
    y=top_actors.index

)

plt.title(
    "Actors With Most Movies"
)

plt.show()

rating_trend = (

    df.groupby("Year")
    ["Rating"]

    .mean()

)

plt.figure(figsize=(12,5))

rating_trend.plot()

plt.title(
    "Rating Trend Over Years"
)

plt.ylabel(
    "Average Rating"
)

plt.show()


trend = (
    df.groupby("Year")
    ["Rating"]
    .mean()
    .reset_index()
)

lr = LinearRegression()

lr.fit(
    trend[["Year"]],
    trend["Rating"]
)

future_years = pd.DataFrame({

    "Year":
    range(
        int(trend["Year"].max())+1,
        int(trend["Year"].max())+6
    )

})

future_years["PredictedRating"] = (
    lr.predict(
        future_years[["Year"]]
    )
)

print(
    future_years
)

plt.figure(figsize=(8,5))

sns.scatterplot(
    x="Duration",
    y="Rating",
    data=df
)

plt.title("Duration vs Rating")

plt.show()


# FEATURE SELECTION


base_features = [

    "Year",
    "Duration",
    "LogVotes",
    "Decade",
    "GenreCount",

    "DirectorFreq",

    "Actor 1_Freq",
    "Actor 2_Freq",
    "Actor 3_Freq",

    "CastPopularity"
]

genre_features = [
    col
    for col in df.columns
    if col.startswith("Genre_")
]

features = base_features + genre_features

X = df[features]

y = df["Rating"]


# TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,
    random_state=42
)


# MODELS
models = {

    "Linear Regression":

    Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "model",
            LinearRegression()
        )
    ]),

    "Random Forest":

    Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=300,
                max_depth=10,
                random_state=42
            )
        )
    ]),

    "XGBoost":

    Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "model",
            XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
        )
    ])
}


# Cross validation

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

print("\nMODEL COMPARISON")
print("="*60)

for name, model in models.items():

    scores = cross_val_score(

        model,
        X_train,
        y_train,

        cv=cv,

        scoring="r2"
    )

    print(
        f"{name:<20} "
        f"R² = {scores.mean():.4f}"
    )

    results.append(
        [name, scores.mean()]
    )


results_df = pd.DataFrame(
    results,
    columns=["Model", "R2"]
)

plt.figure(figsize=(8,5))

sns.barplot(
    x="Model",
    y="R2",
    data=results_df
)

plt.title("Model Comparison")

plt.show()

# Best model

best_model = models["XGBoost"]

best_model.fit(
    X_train,
    y_train
)

predictions = best_model.predict(
    X_test
)


# EVALUATION


mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)

print("\n" + "="*60)
print("XGBOOST RESULTS")
print("="*60)

print("MAE :", round(mae,4))
print("RMSE:", round(rmse,4))
print("R2  :", round(r2,4))


# Actual vs predicted


plt.figure(figsize=(7,6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.5
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    'r--'
)

plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")

plt.title("Actual vs Predicted")
plt.show()

residuals = y_test - predictions

plt.figure(figsize=(8,5))

sns.scatterplot(
    x=predictions,
    y=residuals
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.title("Residual Plot")

plt.xlabel("Predicted Rating")
plt.ylabel("Residuals")

plt.show()


# Feature importance

xgb_model = best_model.named_steps["model"]

importance = pd.DataFrame({

    "Feature": features,

    "Importance":
    xgb_model.feature_importances_

})

importance = (
    importance
    .sort_values(
        by="Importance",
        ascending=False
    )
    .head(15)
)

plt.figure(figsize=(10,6))

sns.barplot(

    x="Importance",
    y="Feature",

    data=importance
)

plt.title(
    "Top 15 Important Features"
)

plt.show()


# Sample prediction
sample = pd.DataFrame({

    "Actual": y_test.values[:10],

    "Predicted":
    predictions[:10]
})

print("\nSample Predictions")
print(sample)


# Save model
with open(
    "movie_rating_model.pkl",
    "wb"
) as f:

    pickle.dump(
        best_model,
        f
    )

print("\nModel Saved Successfully")
