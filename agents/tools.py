"""
SQL tool for cicd.gold schema - safety + schema awareness
"""

# spark is injected from the notebook after loading this module via importlib:
#   import tools_module; tools_module.spark = spark
spark = None


def sql_tool(sql_query: str) -> str:
    """
    Execute SQL on cicd.gold.fact_sale + dim_date ONLY.
    Returns first numeric result.

    Allowed tables ONLY:
    - cicd.gold.fact_sale
    - cicd.gold.dim_date
    """
    if spark is None:
        return "ERROR: spark not injected into tools module."

    # Safety: block DML keywords
    forbidden_keywords = ["insert", "update", "delete", "drop", "truncate", "alter", "create"]
    sql_lower = sql_query.lower()
    for kw in forbidden_keywords:
        if kw in sql_lower:
            return f"ERROR: '{kw.upper()}' statements are not allowed. SELECT only."

    # Safety: only allowed tables
    allowed_tables = ["cicd.gold.fact_sale", "cicd.gold.dim_date"]
    if not any(table in sql_query for table in allowed_tables):
        return "ERROR: Query must reference cicd.gold.fact_sale or cicd.gold.dim_date"

    try:
        df = spark.sql(sql_query)
        result = df.collect()[0][0]
        return f"{result}"
    except Exception as e:
        return f"SQL ERROR: {str(e)}"
