import git

class ModelInstance:
    """Model Instance is a class that stores all the information relative to a model.
    Is the interface that allows to either store a model with Model Governance or reuse a previously run model
    """
    # all attributes stored in one dictionary
    # dict keys are prefixed to reflect which group they belong to:
    # PREPROCESSING_: preprocessing, TRANSFORM_: bagtransform, GENERATOR_: baggenerator, MODEL_: model, OTHER_: comment, hash
    # (these modified indices can be stored easily in csv column header)
    # 2 ways to input a modelinstance: add_info (usually after running a model), add_info_from_csv_line (usually when loading an old model)
    def __init__(self):
        self.dict_attributes = {}

    def get_compressed_info(self):
        return self.dict_attributes

    def update_comment(self, new_comment):
        self.dict_attributes['OTHER_comment'] = new_comment

    def add_info_from_csv_line(self, csv_line_dict):
        self.add_info(self.get_info_by_group(csv_line_dict, "PREPROCESSING"),
                      self.get_info_by_group(csv_line_dict, "TRANSFORM"),
                      self.get_info_by_group(csv_line_dict, "GENERATOR"),
                      self.get_info_by_group(csv_line_dict, "MODEL"),
                      csv_line_dict["OTHER_comment"])

    def add_info(self, preprocessing_attr, bagtransform_attr, baggenerator_attr, model_attr, comment):

        def prefix_dict(d, prefix):
            return {prefix + str(key): val for key, val in d.items()}

        self.dict_attributes = {}

        self.dict_attributes.update(prefix_dict(preprocessing_attr, "PREPROCESSING_"))
        self.dict_attributes.update(prefix_dict(bagtransform_attr, "TRANSFORM_"))
        self.dict_attributes.update(prefix_dict(baggenerator_attr, "GENERATOR_"))
        self.dict_attributes.update(prefix_dict(model_attr, "MODEL_"))
        self.dict_attributes["OTHER_comment"] = comment
        # current hash will be overwritten:
        # repo = git.Repo(search_parent_directories=True)
        # self.dict_attributes["OTHER_git hash"] = repo.head.object.hexsha

    def get_info_by_group(self, input_dict=None, prefix_select=None):
        if input_dict is None:
            input_dict=self.dict_attributes
        new_keys = []
        prefixes = []
        for key, value in input_dict.items():
            prefix, new_key = key.split("_", 1)
            new_keys.append(new_key)
            prefixes.append(prefix)
        input_dict = {new_keys[i]: value for i, value in enumerate(input_dict.values())}
        if prefix_select is None:
            return input_dict
        else:
            return {list(input_dict.keys())[i]:list(input_dict.values())[i] for i, p in enumerate(prefixes) if p==prefix_select}

    def get_baggenerator_info(self):
        # Bag generator also needs some repeated info used in the preprocessing (namely the name of the csv)
        return self.baggenerator_attributes.update({'name_csv': self.preprocessing_attributes['name_csv']})
