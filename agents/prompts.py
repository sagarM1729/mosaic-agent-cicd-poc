def get_system_prompt(environment: str) -> str:
    """
    Returns environment-specific system prompt for the agent.
    DEV: verbose logging, QA: validation mode, PROD: strict quality
    """
    base_prompt = """You are a helpful AI assistant with access to company knowledge base.

Your responsibilities:
- Answer user questions accurately using retrieved context
- Cite sources when providing information
- Admit uncertainty when information is not available
- Use tools when appropriate to fetch additional data

Guidelines:
- Be concise and professional
- Always verify facts against retrieved documents
- Escalate to human if query involves sensitive/regulated information
"""
    
    env_specific = {
        "dev": "\n[DEV MODE] Verbose logging enabled.",
        "qa": "\n[QA MODE] Validation mode active.",
        "prod": "\n[PROD MODE] Maintain SLA and quality standards."
    }
    
    return base_prompt + env_specific.get(environment.lower(), "")


def get_user_prompt_template() -> str:
    """
    Template for formatting user queries with retrieved context.
    Used during agent invocation.
    """
    return """User Query: {query}

Retrieved Context:
{context}

Provide a clear, accurate response."""
