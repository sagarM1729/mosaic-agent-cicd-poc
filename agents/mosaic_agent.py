import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from mlflow.deployments import get_deploy_client
import time
from typing import Dict, Any
from .prompts import get_system_prompt
from .tools import get_agent_tools


class AgentDeployment:

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workspace = WorkspaceClient()
        self.deploy_client = get_deploy_client("databricks")

    def create_agent(self):
        # Logs config + prompts as MLflow artifacts only.
        # No LangChain serialization, no LLM calls during logging.
        system_prompt = get_system_prompt(self.config["environment"])
        tools = get_agent_tools(self.config)

        agent_config = {
            "llm_endpoint": self.config["llm_endpoint"],
            "temperature": self.config.get("temperature", 0.1),
            "max_tokens": self.config.get("max_tokens", 1500),
            "system_prompt": system_prompt,
            "tools": [t["name"] for t in tools] if tools else []
        }

        mlflow.set_experiment(self.config["mlflow_experiment"])

        with mlflow.start_run(run_name=f"agent_{self.config['environment']}") as run:
            mlflow.log_params({
                k: v for k, v in self.config.items()
                if isinstance(v, (str, int, float, bool))
            })
            mlflow.log_dict(agent_config, "agent_config.json")
            mlflow.log_text(system_prompt, "system_prompt.txt")
            print(f"Agent config logged to run: {run.info.run_id}")
            return run

    def deploy_to_serving(self, model_uri: str, endpoint_name: str):
        # Creates or updates Model Serving endpoint.
        endpoint_config = EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=model_uri,
                    scale_to_zero_enabled=True,
                    workload_size=self.config.get("workload_size", "Small"),
                    environment_vars={
                        "DATABRICKS_ENVIRONMENT": self.config["environment"]
                    }
                )
            ]
        )
        try:
            self.workspace.serving_endpoints.create(name=endpoint_name, config=endpoint_config)
            print(f"Created new endpoint: {endpoint_name}")
        except Exception as e:
            if "already exists" in str(e):
                self.workspace.serving_endpoints.update_config(
                    name=endpoint_name,
                    served_entities=endpoint_config.served_entities
                )
                print(f"Updated existing endpoint: {endpoint_name}")
            else:
                raise e
        return endpoint_name

    def wait_for_endpoint(self, endpoint_name: str, timeout: int = 900):
        # Polls until endpoint reaches READY state.
        start_time = time.time()
        while time.time() - start_time < timeout:
            endpoint = self.workspace.serving_endpoints.get(endpoint_name)
            state = endpoint.state.config_update
            if state == "NOT_UPDATING" and endpoint.state.ready == "READY":
                print(f"Endpoint {endpoint_name} is ready")
                return
            print(f"Endpoint state: {state}. Waiting...")
            time.sleep(30)
        raise TimeoutError(f"Endpoint {endpoint_name} not ready after {timeout}s")

    def get_endpoint_url(self, endpoint_name: str):
        host = self.workspace.config.host
        return f"{host}/serving-endpoints/{endpoint_name}/invocations"

    def query_agent(self, endpoint_name: str, user_message: str):
        response = self.workspace.serving_endpoints.query(
            name=endpoint_name,
            inputs={"messages": [{"role": "user", "content": user_message}]}
        )
        return response

    def get_endpoint_metrics(self, endpoint_name: str):
        endpoint = self.workspace.serving_endpoints.get(endpoint_name)
        return {
            "state": endpoint.state.config_update,
            "ready": endpoint.state.ready,
            "pending_update": endpoint.pending_config is not None
        }

    def rollback_endpoint(self, endpoint_name: str, model_version: int):
        # Rolls back to a specific registered model version.
        model_uri = f"models:/{self.config['model_name']}/{model_version}"
        endpoint_config = EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=model_uri,
                    scale_to_zero_enabled=True,
                    workload_size=self.config.get("workload_size", "Small"),
                    environment_vars={"DATABRICKS_ENVIRONMENT": self.config["environment"]}
                )
            ]
        )
        self.workspace.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=endpoint_config.served_entities
        )
        print(f"Rolled back {endpoint_name} to version {model_version}")
        return model_uri

    def delete_endpoint(self, endpoint_name: str):
        self.workspace.serving_endpoints.delete(endpoint_name)
        print(f"Deleted endpoint: {endpoint_name}")
