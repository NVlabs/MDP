import json


class PruneConfigReader(object):
    """
        _conv_gate: dict, key is the conv layer name, value the corresponding gate layer name
        groups: list, each item of this list is a list of names that forms a regularization group
    """
    def __init__(self):
        self._conv_bn = {}
        self._conv_gate = {}
        self.groups = []

    # def _read_field_value(self, key, default):
    #     value = default
    #     if key in self.method_config:
    #         value = self.method_config[key]
    #     self.prune_setting[key] = value

    def set_prune_setting(self, param_config):
        """
            Set the regularization to the layers according to the configuration file
        Args:
            param_config:
        """
        with open(param_config, 'r') as f:
            cf_dict = json.load(f)
        print(cf_dict)

        reg_groups = cf_dict['reg_groups']
        for group in reg_groups:
            reg_type = group['reg_type']
            assert reg_type in ['GS_SPARSE', 'CL_GROUP']
            for item in group['layers']:
                name_list = item['layer_name'].replace(' ', '').split(',')
                bn_name_list = (item['bn_name'].replace(' ', '').split(',')
                                if item['bn_name']
                                else ['' for _ in range(len(name_list))])
                gate_name_list = (item['gate_name'].replace(' ', '').split(',')
                                  if item['gate_name']
                                  else ['' for _ in range(len(name_list))])
                if len(gate_name_list) == 1 and reg_type == 'CL_GROUP':
                    gate_name_list = gate_name_list * len(name_list)
                assert len(gate_name_list) == len(name_list)
                assert len(bn_name_list) == len(name_list)
                for name, bn_name, gate_name in zip(name_list, bn_name_list, gate_name_list):
                    self._conv_bn[name] = bn_name
                    self._conv_gate[name] = gate_name

                if reg_type == 'GS_SPARSE':
                    self.groups.extend([tuple([name]) for name in name_list])
                else:
                    self.groups.append(tuple(name_list))

    def get_layer_structure(self):
        has_bn, has_gate = False, False
        for k, v in self._conv_bn.items():
            if v != '':
                has_bn = True
                break
        for k, v in self._conv_gate.items():
            if v != '':
                has_gate = True
                break
        if not has_bn:
            self._conv_bn = None
        if not has_gate:
            self._conv_gate = None
        return self._conv_bn, self._conv_gate, self.groups
