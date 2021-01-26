#%%
import mlflow
from mlflow.tracking.client import MlflowClient
from mlflow.entities import ViewType
#%%
def print_run_info(run_info):        
    print("- run_id: {}, lifecycle_stage: {}".format(run_info.run_id, run_info.lifecycle_stage))

def search_runs(cfgfile):
    query = "params.cfgname='%s'"
    runs = MlflowClient().search_runs(experiment_ids="0", 
        filter_string=query % cfgfile, 
        run_view_type=ViewType.ACTIVE_ONLY)

    runs_win = MlflowClient().search_runs(experiment_ids="0", 
        filter_string=query % cfgfile.replace('/','\\'), 
        run_view_type=ViewType.ACTIVE_ONLY)
    
    for run in runs+runs_win:        
        print_run_info(run.info)
        print(run.data)
#%%
if __name__=="__main__":
    
    search_runs('exploration/monai_abc.json')

    # print("Active runs:")
    # print_run_infos(mlflow.list_run_infos("0", run_view_type=ViewType.ACTIVE_ONLY))

    # print("Deleted runs:")
    # print_run_infos(mlflow.list_run_infos("0", run_view_type=ViewType.DELETED_ONLY))

    # print("All runs:")
    # print_run_infos(mlflow.list_run_infos("0", run_view_type=ViewType.ALL))