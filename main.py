import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from ExplanationEvaluation.configs.selector import Selector
from ExplanationEvaluation.tasks.replication import replication

dataset = 'mutag'
explainer_name = 'proxy'
folder = 'replication'
config_path = f"./ExplanationEvaluation/configs/{folder}/explainers/{explainer_name}/{dataset}.json"
config = Selector(config_path)

is_extension = (folder == 'extension')


(auc, auc_std), inference_time = replication(
    config.args.explainer,
    is_extension,
    device='cuda:0'
)

print((auc, auc_std), inference_time)