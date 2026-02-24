# Databricks notebook source
# USE EXACT FILENAMES from your Volume
fact_sale = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv("/Volumes/cicd/gold/uploads/gold_fact.csv")  # ← fixed name

dim_date = spark.read \
    .option("header", True) \
    .option("inferSchema", True) \
    .csv("/Volumes/cicd/gold/uploads/dim_date.csv")   # ← this was already correct

# Save as Delta tables
fact_sale.write.format("delta").mode("overwrite").saveAsTable("cicd.gold.fact_sale")
dim_date.write.format("delta").mode("overwrite").saveAsTable("cicd.gold.dim_date")

print(f"fact_sale: {fact_sale.count()} rows ✅")
print(f"dim_date : {dim_date.count()} rows ✅")


# COMMAND ----------

# MAGIC %sql
# MAGIC -- fact_sale columns
# MAGIC DESCRIBE TABLE cicd.gold.fact_sale;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- dim_date columns  
# MAGIC DESCRIBE TABLE cicd.gold.dim_date;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT 
# MAGIC   d.calendar_year,
# MAGIC   d.calendar_month_label,
# MAGIC   d.calendar_month_number,
# MAGIC   COUNT(DISTINCT wwi_invoice_id) AS invoice_count,
# MAGIC   COUNT(*) AS line_items,
# MAGIC   ROUND(SUM(total_including_tax), 0) AS total_sales
# MAGIC FROM cicd.gold.fact_sale fs
# MAGIC JOIN cicd.gold.dim_date d ON fs.invoice_date_key = d.date_key
# MAGIC GROUP BY d.calendar_year, d.calendar_month_label, d.calendar_month_number
# MAGIC ORDER BY d.calendar_year, d.calendar_month_label
# MAGIC