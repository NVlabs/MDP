import json


class PruneConfigReader(object):
    """
        _conv_gate: dict, key is the conv layer name, value the corresponding gate layer name
        _conv_bn: dict, key is the conv layer name, value the corresponding batch norm layer name
        groups: list, each item of this list is a list of names that forms a regularization group
        is_main: indicator if the class is part of the main thread
    """
    def __init__(self, is_main=True):
        self._conv_bn = {}
        self._conv_gate = {}
        self.groups = []
        self.is_main = is_main

    def set_prune_setting(self, param_config):
        """
            Set the regularization to the layers according to the configuration file
        Args:
            param_config: the configuration json file that contains the structure of the model
        """
        with open(param_config, 'r') as f:
            cf_dict = json.load(f)
        if self.is_main:
            print(cf_dict)

        for item in cf_dict['layers']:
            name_list = item['layer_name'].replace(' ', '').split(',')
            bn_name_list = (item['bn_name'].replace(' ', '').split(',')
                            if item['bn_name']
                            else ['' for _ in range(len(name_list))])
            gate_name_list = (item['gate_name'].replace(' ', '').split(',')
                              if item['gate_name']
                              else ['' for _ in range(len(name_list))])
            if len(gate_name_list) == 1:
                gate_name_list = gate_name_list * len(name_list)
            #assert len(gate_name_list) == len(name_list)
            #assert len(bn_name_list) == len(name_list)
            for name, bn_name, gate_name in zip(name_list, bn_name_list, gate_name_list):
                self._conv_bn[name] = bn_name
                self._conv_gate[name] = gate_name
                self.groups.append(name)

    def get_layer_structure(self):
        return self._conv_bn, self._conv_gate, self.groups
