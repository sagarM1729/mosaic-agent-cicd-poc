from typing import List, Dict, Any
from datetime import datetime


def get_current_date():
    """
    Returns current timestamp.
    Used by agent for time-sensitive queries.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def query_database(sql: str = "", catalog: str = "main", schema: str = "default"):
    """
    Executes SQL against Unity Catalog.
    Requires enable_sql_tool=true in config.
    Returns: dict with results or error
    """
    if not sql:
        return {"error": "SQL query is required"}
    
    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        
        spark.sql(f"USE CATALOG {catalog}")
        spark.sql(f"USE SCHEMA {schema}")
        df = spark.sql(sql)
        return df.toPandas().to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}


def get_agent_tools(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns list of tools based on config flags.
    Each tool = {name, description, function, parameters}
    """
    tools = []
    
    if config.get("enable_date_tool", True):
        tools.append({
            "name": "get_current_date",
            "description": "Returns the current date and time",
            "function": get_current_date,
            "parameters": {}
        })
    
    if config.get("enable_sql_tool", False):
        tools.append({
            "name": "query_database",
            "description": "Execute SQL query against Unity Catalog tables",
            "function": query_database,
            "parameters": {
                "sql": {"type": "string", "description": "SQL query to execute"},
                "catalog": {"type": "string", "description": "UC catalog name"},
                "schema": {"type": "string", "description": "UC schema name"}
            }
        })
    
    return tools if tools else None
