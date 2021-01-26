#%%
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
#%%
# %reset -f
import torch
import mlflow
import mlflow.pytorch
from torchsummary import summary
from pipeline import monai_utils
from monai.handlers import StatsHandler
from mlflow.tracking import MlflowClient
from pipeline.monai_utils import parseInputs
#%%
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
#%%
def print_auto_logged_info(r):
    tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
    artifacts = [f.path for f in MlflowClient().list_artifacts(r.info.run_id, "model")]
    print("run_id: {}".format(r.info.run_id))
    print("artifacts: {}".format(artifacts))
    print("params: {}".format(r.data.params))
    print("metrics: {}".format(r.data.metrics))
    print("tags: {}".format(tags))
#%%
def train(cfg, overrides=None):
    with mlflow.start_run() as run:
        if overrides is not None:
            cfg.update_from_args(overrides)
        mlflow.log_params(cfg.config)
        (train_dataset,valid_dataset) = cfg.get_dataset()
        (train_loader,valid_loader) = cfg.get_loaders(train_dataset,valid_dataset)
        my_model = cfg.get_model_instance()
        my_model = my_model.to(device)
        opt = cfg.get_optimizer_instance(my_model)
        loss_func = cfg.get_criterion_instance()
#         summary(my_model,(1,128,128,128))
        inferer = None
        train_handlers = StatsHandler(tag_name="train_loss", output_transform=lambda x: x["loss"]),
        trainer = cfg.get_train_engine(device,train_loader,valid_loader,my_model,inferer,opt,loss_func,train_handlers)
        trainer.run()
        mlflow.pytorch.log_model(my_model, "model")
#         print(run.info, run.data)
        print_auto_logged_info(mlflow.get_run(run_id = run.info.run_id))
#%%
cfg, my_args = parseInputs()
#%%
train(cfg = cfg,overrides = my_args)
# %%
