
from mlflow.tracking import MlflowClient
import mlflow.pytorch




def load_model_by_devaddr(devaddr, client):
    runs = client.search_runs(
        experiment_ids=["0"],
        filter_string=f"tags.DevAddr = '{devaddr}' and tags.Stage = 'production'",
        order_by=["metrics.accuracy DESC"],
        max_results=1
    )

    if not runs:
        return None

    run_id = runs[0].info.run_id
    model_uri = f"runs:/{run_id}/model"
    model = mlflow.pytorch.load_model(model_uri)
    return model