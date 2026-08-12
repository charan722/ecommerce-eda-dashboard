# src/data_loader.py
import os
import pandas as pd
import numpy as np

RAW_DATA_PATH = "data/Online Retail.xlsx"
PROCESSED_DATA_PATH = "data/processed_retail.parquet"

def load_raw_data(file_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Reads raw Excel file from the dataset directory."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"File not found at {file_path}. Please place 'Online Retail.xlsx' in the data/ folder."
        )
    print("Reading raw excel file (this may take a few seconds)...")
    return pd.read_excel(file_path)


def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw retail data:
    1. Removes duplicates
    2. Flags and separates cancelled orders (InvoiceNo starting with 'C')
    3. Filters out bad/negative quantities and zero unit prices
    4. Imputes missing CustomerIDs and Descriptions
    5. Engineers financial and temporal columns
    """
    df_clean = df.copy()

    # 1. Deduplication
    df_clean.drop_duplicates(inplace=True)

    # 2. Handle Cancelled Transactions
    df_clean["InvoiceNo"] = df_clean["InvoiceNo"].astype(str)
    df_clean["IsCancelled"] = df_clean["InvoiceNo"].str.startswith("C")

    # 3. Filter valid sales records (Positive Quantity & Price)
    # Note: Keep non-cancelled rows for primary sales analytics
    df_clean = df_clean[~df_clean["IsCancelled"]]
    df_clean = df_clean[(df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)]

    # 4. Text and Categorical Cleaning
    df_clean["Description"] = (
        df_clean["Description"].astype(str).str.strip().str.upper()
    )
    df_clean["Description"] = df_clean["Description"].replace({"NAN": "UNKNOWN ITEM"})
    df_clean["StockCode"] = df_clean["StockCode"].astype(str)

    # 5. Missing Value Imputation for CustomerID
    # Instead of dropping, assign -1 to track guest/unregistered users
    df_clean["CustomerID"] = (
        df_clean["CustomerID"].fillna(-1).astype(int)
    )

    # 6. Feature Engineering (Metrics & Datetime Breakdown)
    df_clean["TotalAmount"] = df_clean["Quantity"] * df_clean["UnitPrice"]
    df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
    df_clean["YearMonth"] = df_clean["InvoiceDate"].dt.to_period("M").astype(str)
    df_clean["Hour"] = df_clean["InvoiceDate"].dt.hour
    df_clean["DayOfWeek"] = df_clean["InvoiceDate"].dt.day_name()

    return df_clean


def get_data(force_reload: bool = False) -> pd.DataFrame:
    """
    Master ingestion function with caching. Converts cleaned data to Parquet
    for 10x-20x faster read performance in Streamlit.
    """
    if os.path.exists(PROCESSED_DATA_PATH) and not force_reload:
        print("Loading cached dataset from Parquet...")
        return pd.read_parquet(PROCESSED_DATA_PATH)

    print("Processing dataset from raw file...")
    raw_df = load_raw_data()
    cleaned_df = transform_data(raw_df)

    # Save to parquet
    cleaned_df.to_parquet(PROCESSED_DATA_PATH, index=False)
    print(f"Data saved successfully to {PROCESSED_DATA_PATH}!")
    
    return cleaned_df

if __name__ == "__main__":
    # Test script locally
    df = get_data(force_reload=True)
    print("\n--- INGESTION SUMMARY ---")
    print(f"Dataset Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Total Revenue: ${df['TotalAmount'].sum():,.2f}")
    print(f"Unique Customers: {df[df['CustomerID'] != -1]['CustomerID'].nunique():,}")
