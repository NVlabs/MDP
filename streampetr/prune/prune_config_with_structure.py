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
        self._pre_group = {}
        self._aft_group_list = {}
        self._name_to_group = {}

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
        # print(cf_dict)

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
                pre_conv_name_list = (item['pre_conv'].replace(' ', '').split(',')
                                      if item['pre_conv'] else ['' for _ in range(len(name_list))])
                aft_conv_name_list = (item['aft_conv'].replace(' ', '').split(',')
                                      if item['aft_conv'] else [])
                if len(gate_name_list) == 1 and reg_type == 'CL_GROUP':
                    gate_name_list = gate_name_list * len(name_list)
                assert len(gate_name_list) == len(name_list)
                assert len(bn_name_list) == len(name_list)
                assert len(pre_conv_name_list) == len(name_list)
                for name, bn_name, gate_name in zip(name_list, bn_name_list, gate_name_list):
                    self._conv_bn[name] = bn_name
                    self._conv_gate[name] = gate_name

                if reg_type == 'GS_SPARSE':
                    self.groups.extend([tuple([name]) for name in name_list])
                    assert (name_list[0] == 'module.conv1' or name_list[0] == 'module.features.conv_dw13.conv2' or 
                            name_list[0] == 'module.features.conv5-3' or name_list[0] == 'module.features.conv4-3' or 
                            len(aft_conv_name_list) == len(name_list))
                    if len(name_list) == 1 and name_list[0] == 'module.conv1':
                        self._aft_group_list[tuple(name_list)] = aft_conv_name_list
                    else:
                        for name, aft_conv_name in zip(name_list, aft_conv_name_list):
                            self._aft_group_list[tuple([name])] = [aft_conv_name]
                    for name, pre_conv_name in zip(name_list, pre_conv_name_list):
                        self._name_to_group[name] = tuple([name])
                else:
                    self.groups.append(tuple(name_list))
                    for name in name_list:
                        self._name_to_group[name] = self.groups[-1]
                    self._aft_group_list[tuple(name_list)] = aft_conv_name_list

                for name, pre_conv_name in zip(name_list, pre_conv_name_list):
                    self._pre_group[name] = pre_conv_name

        for group_name, pre_conv_name in self._pre_group.items():
            self._pre_group[group_name] = self._name_to_group[pre_conv_name] if pre_conv_name in self._name_to_group else None
        print(self._aft_group_list)
        for group_name, aft_conv_list in self._aft_group_list.items():
            for ind, aft_conv_name in enumerate(aft_conv_list):
                aft_conv_list[ind] = self._name_to_group[aft_conv_name]
            self._aft_group_list[group_name] = aft_conv_list

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
        return self._conv_bn, self._conv_gate, self.groups, self._pre_group, self._aft_group_list
