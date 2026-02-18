import yaml
import sys
import mlflow
from agents.mosaic_agent import AgentDeployment


def load_config(env: str):
    config_path = f"../config/{env}.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def deploy_agent(env: str):
    print(f"Starting deployment for environment: {env}")

    config = load_config(env)
    deployment = AgentDeployment(config)

    print("Creating agent...")
    run = deployment.create_agent()
    run_id = run.info.run_id
    print(f"Agent logged with run_id: {run_id}")

    print(f"\nDeployment Phase 1 complete.")
    print(f"MLflow Experiment: {config['mlflow_experiment']}")
    print(f"Run ID: {run_id}")

    return {
        "run_id": run_id,
        "experiment": config["mlflow_experiment"],
        "environment": env
    }


if __name__ == "__main__":
    env = sys.argv[1] if len(sys.argv) > 1 else "dev"
    if env not in ["dev", "qa", "prod"]:
        print(f"Invalid environment: {env}")
        sys.exit(1)
    try:
        result = deploy_agent(env)
        print(result)
    except Exception as e:
        print(f"Deployment failed: {str(e)}")
        sys.exit(1)