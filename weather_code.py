import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------- CONFIG -------------
DATA_PATH = "data/DailyDelhiClimateTrain.csv" # Kaggle file
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------- TASK 1: LOAD & INSPECT DATA -------------
def load_data(path: str) -> pd.DataFrame:
    """Load the weather data from CSV."""
    df = pd.read_csv(path)
    print("=== Raw Data Head ===")
    print(df.head())
    print("\n=== Raw Data Info ===")
    print(df.info())
    print("\n=== Raw Data Describe ===")
    print(df.describe())
    return df


# ------------- TASK 2: CLEANING & PROCESSING -------------
def clean_data(df):
    # parse date safely
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df = df[["date", "temp", "humidity", "wind", "pressure", "rainfall"]]

    df = df.dropna()
    df = df.sort_values("date")

    print("\n=== Cleaned Data Head ===")
    print(df.head())
    print("\n=== Cleaned Data Info ===")
    print(df.info())

    return df


    # Convert date column to datetime
    df["date"] = pd.to_datetime(df["date"])

    df = df[["date", "temp", "humidity"]]

    df = df.dropna(subset=["temp", "humidity"])

    df = df.sort_values("date")

    print("\n=== Cleaned Data Head ===")
    print(df.head())
    print("\n=== Cleaned Data Info ===")
    print(df.info())

    return df


# ------------- TASK 3: STATISTICAL ANALYSIS WITH NUMPY -------------
def compute_statistics(df: pd.DataFrame):
    """
    Compute:
    - Daily: describe() as basic summary
    - Monthly: mean, min, max, std
    - Yearly: mean, min, max, std using NumPy
    """
    print("\n=== Daily Level Summary (describe) ===")
    daily_stats = df[["temp", "humidity"]].describe()
    print(daily_stats)

    # Monthly stats
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    monthly_stats = (
        df.groupby(["year", "month"])[["temp", "humidity"]]
        .agg(["mean", "min", "max", "std"])
        .round(2)
    )
    print("\n=== Monthly Statistics (mean, min, max, std) ===")
    print(monthly_stats)

    temps = df["temp"].values
    hums = df["humidity"].values

    yearly_stats = {
        "temp_mean": float(np.mean(temps)),
        "temp_min": float(np.min(temps)),
        "temp_max": float(np.max(temps)),
        "temp_std": float(np.std(temps)),
        "humidity_mean": float(np.mean(hums)),
        "humidity_min": float(np.min(hums)),
        "humidity_max": float(np.max(hums)),
        "humidity_std": float(np.std(hums)),
    }

    print("\n=== Yearly Statistics using NumPy ===")
    for k, v in yearly_stats.items():
        print(f"{k}: {v:.2f}")

    return daily_stats, monthly_stats, yearly_stats


# ------------- TASK 4: VISUALIZATION WITH MATPLOTLIB -------------
def plot_daily_temperature(df: pd.DataFrame, out_dir: str):
    """Line chart for daily temperature trends."""
    plt.figure(figsize=(10, 5))
    plt.plot(df["date"], df["temp"])
    plt.xlabel("Date")
    plt.ylabel("Temperature (°C)")
    plt.title("Daily Temperature Trend")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "daily_temperature.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_monthly_avg_wind(df: pd.DataFrame, out_dir: str):
    """Line chart for monthly average wind speed."""
    df["year_month"] = df["date"].dt.to_period("M")
    monthly_wind = df.groupby("year_month")["wind"].mean()

    plt.figure(figsize=(8,5))
    monthly_wind.plot(marker="o", linestyle="-")
    plt.xlabel("Month")
    plt.ylabel("Wind Speed (km/h)")
    plt.title("Monthly Average Wind Speed")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "monthly_avg_wind.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")



def plot_monthly_avg_temperature(df: pd.DataFrame, out_dir: str):
    """Bar chart for monthly average temperature (instead of rainfall)."""
    df["year_month"] = df["date"].dt.to_period("M")
    monthly_avg_temp = df.groupby("year_month")["temp"].mean()

    plt.figure(figsize=(8, 5))
    monthly_avg_temp.plot(kind="bar")
    plt.xlabel("Month")
    plt.ylabel("Average Temperature (°C)")
    plt.title("Monthly Average Temperature")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "monthly_avg_temperature.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")

def plot_monthly_rainfall(df: pd.DataFrame, out_dir: str):
    """Bar chart for monthly total rainfall."""
    df["year_month"] = df["date"].dt.to_period("M")
    monthly_rainfall = df.groupby("year_month")["rainfall"].sum()

    plt.figure(figsize=(8,5))
    monthly_rainfall.plot(kind="bar")
    plt.xlabel("Month")
    plt.ylabel("Total Rainfall (mm)")
    plt.title("Monthly Rainfall Totals")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "monthly_rainfall.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_humidity_vs_temp(df: pd.DataFrame, out_dir: str):
    """Scatter plot for humidity vs temperature."""
    plt.figure(figsize=(7, 5))
    plt.scatter(df["temp"], df["humidity"])
    plt.xlabel("Temperature (°C)")
    plt.ylabel("Humidity (%)")
    plt.title("Humidity vs Temperature")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "humidity_temp_scatter.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


def plot_combined_figure(df: pd.DataFrame, out_dir: str):
    """Combined figure with two plots."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    # Top: temperature line
    axes[0].plot(df["date"], df["temp"])
    axes[0].set_title("Daily Temperature Trend")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Temperature (°C)")

    # Bottom: humidity vs temperature scatter
    axes[1].scatter(df["temp"], df["humidity"])
    axes[1].set_title("Humidity vs Temperature")
    axes[1].set_xlabel("Temperature (°C)")
    axes[1].set_ylabel("Humidity (%)")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "combined_plots.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved: {out_path}")


# ------------- TASK 5: GROUPING & AGGREGATION -------------
def group_and_aggregate(df: pd.DataFrame):
    """
    Group data by month and calculate aggregate statistics.
    """
    df["year_month"] = df["date"].dt.to_period("M")
    grouped = (
        df.groupby("year_month")[["temp", "humidity"]]
        .agg(["mean", "min", "max", "std"])
        .round(2)
    )

    print("\n=== Grouped by Month (Aggregated Stats) ===")
    print(grouped)
    return grouped


# ------------- TASK 6: EXPORT CLEANED DATA -------------
def export_cleaned_data(df: pd.DataFrame, out_dir: str):
    """Export cleaned data to CSV."""
    out_path = os.path.join(out_dir, "cleaned_weather.csv")
    df.to_csv(out_path, index=False)
    print(f"\nCleaned data saved to: {out_path}")


# ------------- MAIN DRIVER -------------
def main():
    # Load
    df_raw = load_data(DATA_PATH)

    # Clean
    df_clean = clean_data(df_raw)

    # Stats
    compute_statistics(df_clean)

    # Grouping and aggregation
    group_and_aggregate(df_clean)

    # Visualizations
    plot_daily_temperature(df_clean, OUTPUT_DIR)
    plot_monthly_avg_temperature(df_clean, OUTPUT_DIR)
    plot_humidity_vs_temp(df_clean, OUTPUT_DIR)
    plot_combined_figure(df_clean, OUTPUT_DIR)
    plot_monthly_rainfall(df_clean, OUTPUT_DIR)
    plot_monthly_avg_wind(df_clean, OUTPUT_DIR)



    # Export cleaned data
    export_cleaned_data(df_clean, OUTPUT_DIR)

    print("\n=== Analysis Complete! ===")

    


if __name__ == "__main__":
    main()
