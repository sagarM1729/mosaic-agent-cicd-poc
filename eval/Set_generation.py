# Databricks notebook source
import pandas as pd

# 1. NEW JULY GOLDEN QUESTIONS + ANSWERS
# We provide 30 questions covering counts, sums, and string labels.
# This satisfies the "Evaluation-First" requirement of the CI/CD pipeline.
golden_data = [
    {"question": "What is the total sales quantity in July?", "expected_answer": "116830"},
    {"question": "What is the total profit in July?", "expected_answer": "2902877.0"},
    {"question": "What is the highest unit price in July?", "expected_answer": "345.0"},
    {"question": "What is the lowest unit price in July?", "expected_answer": "5.0"},
    {"question": "What is the average tax rate in July?", "expected_answer": "15.0"},
    {"question": "What is the total tax amount in July?", "expected_answer": "865950.75"},
    {"question": "What is the most sold item description in July?", "expected_answer": "Furry animal socks (Pink) L"},
    {"question": "What is the total number of invoices in July?", "expected_answer": "12259"},
    {"question": "What is the average profit per sale in July?", "expected_answer": "145.14"},
    {"question": "What is the city with the highest sales in July?", "expected_answer": "275141"},
    {"question": "What is the customer with the most purchases in July?", "expected_answer": "8134"},
    {"question": "What is the most common package type sold in July?", "expected_answer": "Each"},
    {"question": "What is the total number of chiller items sold in July?", "expected_answer": "0"},
    {"question": "What is the total number of dry items sold in July?", "expected_answer": "40171"},
    {"question": "What is the maximum quantity sold in a single sale in July?", "expected_answer": "120"},
    {"question": "What is the minimum quantity sold in a single sale in July?", "expected_answer": "1"},
    {"question": "What is the average total including tax per sale in July?", "expected_answer": "331.95"},
    {"question": "What is the salesperson with the highest sales in July?", "expected_answer": "4525"},
    {"question": "What is the number of unique stock items sold in July?", "expected_answer": "82"},
    {"question": "What is the most frequent invoice date in July?", "expected_answer": "2016-07-30"},
    {"question": "What is the total number of sales transactions in July?", "expected_answer": "20000"},
    {"question": "What is the average unit price in July?", "expected_answer": "52.20"},
    {"question": "What is the most common day of the week for sales in July?", "expected_answer": "Saturday"},
    {"question": "What is the total sales excluding tax in July?", "expected_answer": "5773005.0"},
    {"question": "What is the highest profit from a single sale in July?", "expected_answer": "1515.0"},
    {"question": "What is the lowest profit from a single sale in July?", "expected_answer": "8.5"},
    {"question": "What is the most common customer key in July?", "expected_answer": "8134"},
    {"question": "What is the most common city key in July?", "expected_answer": "248960"},
    {"question": "What is the most common salesperson key in July?", "expected_answer": "4531"},
    {"question": "What is the most common bill_to_customer_key in July?", "expected_answer": "8066"}
]

# 2. DATA NORMALIZATION (Standardize to String)
# Convert to DataFrame and force 'expected_answer' to String.
# This prevents Delta Lake type-mismatch errors when mixed types are present.
golden_df = pd.DataFrame(golden_data)
golden_df['expected_answer'] = golden_df['expected_answer'].astype(str)

# Smoke set is used for rapid CI checks (first 5 questions)
smoke_df = golden_df.head(5)

# 3. SAVE TO VOLUMES (For Git/CSV Portability)
# Note: Ensure path exists: /Volumes/cicd/gold/uploads/
golden_df.to_csv("/Volumes/cicd/gold/uploads/golden_set.csv", index=False)
smoke_df.to_csv("/Volumes/cicd/gold/uploads/smoke_set.csv", index=False)

# 4. SAVE TO DELTA (The "Source of Truth" for test.py)
# We use .option("overwriteSchema", "true") to fix the DELTA_FAILED_TO_MERGE_FIELDS error.
# This ensures the 'expected_answer' column changes from FLOAT/INT to STRING safely.
spark.createDataFrame(golden_df).write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("cicd.gold.golden_set")

spark.createDataFrame(smoke_df).write \
    .format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .saveAsTable("cicd.gold.smoke_set")

print(f" SUCCESSFULLY GENERATED 30 JULY QUESTIONS")
print(f" Tables cicd.gold.golden_set and smoke_set are now STRING based.")