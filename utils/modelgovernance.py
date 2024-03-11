import pandas as pd
from utils.modelinstance import ModelInstance


class ModelGovernance:
    """ Model Governance is a class that stores models after running along with all its information.
    It also allows the retrieval of previously run models
    """

    PATH_MODELS = "Model Storage/Models/"
    PATH_MODELS_INFO = "Model Storage/models.csv"

    def __init__(self):
        try:
            self.models_info = pd.read_csv(ModelGovernance.PATH_MODELS_INFO, delimiter=";", index_col="index")
        except(FileNotFoundError):
            self.models_info = pd.DataFrame()

    def add_instance(self, modelinstance):
        new_line = pd.DataFrame.from_dict(modelinstance.get_compressed_info(), orient='index').transpose()
        self.models_info = pd.concat([self.models_info, new_line])
        self.models_info = self.models_info.reset_index(drop=True)
        self.models_info.to_csv(ModelGovernance.PATH_MODELS_INFO, sep=";", index_label="index")

    def get_instance(self, i: int):
        new_instance_dict = self.models_info.iloc[i].to_dict()
        mod_inst = ModelInstance()
        mod_inst.add_info_from_csv_line(new_instance_dict)
        return mod_inst

    def update_instance(self, model_instance: ModelInstance, i: int):
        self.models_info.iloc[i] = model_instance.get_compressed_info()
        self.models_info.to_csv(ModelGovernance.PATH_MODELS_INFO, sep=";")

    def get_models(self):
        return self.models_info
