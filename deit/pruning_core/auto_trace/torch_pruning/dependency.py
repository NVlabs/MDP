import torch
import torch.nn as nn
import typing
from functools import reduce
from operator import mul
from enum import IntEnum
import pdb

__all__ = ['Dependency', 'DependencyGraph']

TORCH_CONV = nn.modules.conv._ConvNd
TORCH_TCONV = nn.modules.conv._ConvNd
TORCH_BATCHNORM = nn.modules.batchnorm._BatchNorm
TORCH_PRELU = nn.PReLU
TORCH_LINEAR = nn.Linear

class OPTYPE(IntEnum):
    CONV = 0
    BN = 1
    LINEAR = 2
    PRELU = 3
    GROUP_CONV=4

    CONCAT=5
    SPLIT=6
    ELEMENTWISE=7

def _get_module_type(module):
    if isinstance( module, TORCH_CONV ):
        if module.groups>1:
            return OPTYPE.GROUP_CONV
        else:
            return OPTYPE.CONV
    elif isinstance( module, TORCH_BATCHNORM ):
        return OPTYPE.BN
    elif isinstance( module, TORCH_PRELU ):
        return OPTYPE.PRELU
    elif isinstance( module, TORCH_LINEAR ):
        return OPTYPE.LINEAR
    elif isinstance( module, _ConcatOp ):
        return OPTYPE.CONCAT
    elif isinstance( module, _SplitOP):
        return OPTYPE.SPLIT
    else:
        return OPTYPE.ELEMENTWISE

# Dummy Pruning fn
def _prune_concat(layer, *args, **kargs):
    return layer, 0

def _prune_split(layer, *args, **kargs):
    return layer, 0

def _prune_elementwise_op(layer, *args, **kargs):
    return layer, 0

# Dummy module
class _ConcatOp(nn.Module):
    def __init__(self):
        super(_ConcatOp, self).__init__()
        self.offsets = None
        
    def __repr__(self):
        return "_ConcatOp(%s)"%(self.offsets)

class _SplitOP(nn.Module):
    def __init__(self):
        super(_SplitOP, self).__init__()
        self.offsets = None
        
    def __repr__(self):
        return "_SplitOP(%s)"%(self.offsets)

class _ElementWiseOp(nn.Module):
    def __init__(self):
        super(_ElementWiseOp, self).__init__()

    def __repr__(self):
        return "_ElementWiseOp()"



class _FlattenIndexTransform(object):
    def __init__(self, stride=1, reverse=False):
        self._stride = stride
        self.reverse = reverse

    def __call__(self, idxs):
        new_idxs = []
        if self.reverse==True:
            for i in idxs:
                new_idxs.append( i//self._stride )
                new_idxs = list( set(new_idxs) )
        else:
            for i in idxs:
                new_idxs.extend(list( range(i*self._stride, (i+1)*self._stride)))
        return new_idxs

class _ConcatIndexTransform(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs):
        if self.reverse==True:
            new_idxs = [i-self.offset[0] for i in idxs if (i>=self.offset[0] and i<self.offset[1])]
        else:
            new_idxs = [i+self.offset[0] for i in idxs]
        return new_idxs

class _SplitIndexTransform(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs):
        if self.reverse==True:
            new_idxs = [i+self.offset[0] for i in idxs ]
        else:
            new_idxs = [i-self.offset[0] for i in idxs if (i>=self.offset[0] and i<self.offset[1])]
        return new_idxs
        
class Node(object):
    def __init__(self, module, grad_fn, node_name=None):
        self.module = module
        self.grad_fn = grad_fn
        self.inputs = []
        self.outputs = []
        self.dependencies = []
        self._node_name = node_name 
        self.type = _get_module_type( module )

    @property
    def node_name(self):
        return "%s (%s)"%(self._node_name, str(self.module)) if self._node_name is not None else str(self.module)

    def add_input(self, node):
        if node not in self.inputs:
            self.inputs.append( node )
    
    def add_output(self, node):
        if node not in self.outputs:
            self.outputs.append( node )

    def __repr__(self):
        return "<Node: (%s, %s)>"%( self.node_name, self.grad_fn )

    def __str__(self):
        return "<Node: (%s, %s)>"%( self.node_name, self.grad_fn )

    def details(self):
        fmt = "<Node: (%s, %s)>\n"%( self.node_name, self.grad_fn )
        fmt += ' '*4+'IN:\n'
        for in_node in self.inputs:
            fmt+=' '*8+'%s\n'%(in_node)
        fmt += ' '*4+'OUT:\n'
        for out_node in self.outputs:
            fmt+=' '*8+'%s\n'%(out_node)

        fmt += ' '*4+'DEP:\n'
        for dep in self.dependencies:
            fmt+=' '*8+"%s\n"%(dep)
        return fmt


class Dependency(object):
    def __init__(self, node, type="in", shift=0, transformation = None):
        """ Layer dependency in structed neural network pruning.
            type: input/output
        Parameters:

        """
        self.type = type
        self.node = node
        self.shift = shift
        self.transformation = transformation # will keep track of transformations across layers.

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "<DEP: type: %s, module %s, shift: %s>" % (self.type, self.node, self.shift)

class DependencyGraph(object):

    PRUNABLE_MODULES = ( nn.modules.conv._ConvNd, nn.modules.batchnorm._BatchNorm, nn.Linear)
    
    def build_dependency( self, model:torch.nn.Module, example_inputs:torch.Tensor, output_transform:callable=None, verbose:bool=True ):
        self.verbose = verbose
        # get module name
        self._module_to_name = { module: name for (name, module) in model.named_modules() }
        # build dependency graph
        self.module_to_node = self._obtain_forward_graph( model, example_inputs, output_transform=output_transform )
        self._build_dependency_BN(self.module_to_node)
        # pdb.set_trace()
        return self

    def _build_dependency_BN(self, module_to_node):

        def _get_breakable_input(node, main_node):
            for in_node in node.inputs:
                if in_node.type in BREAKABLE_NODES:
                    dep = Dependency(node=in_node, type="in", shift=0)
                    main_node.dependencies.append(dep)
                else:
                    _get_breakable_input(in_node, main_node)

        def _rec_dependencies_copy(out_node, main_node):
            for dep_internal in out_node.dependencies:
                # if (dep_internal.node.type in BREAKABLE_NODES):
                if dep_internal not in main_node.dependencies:
                    if len(dep_internal.node.dependencies)>0:
                        #go only 1 layer because we keep updating it
                        for dep_internal2 in dep_internal.node.dependencies:
                            check_existence = any([dep_internal2.node == a.node for a in main_node.dependencies])
                            if not check_existence:
                                main_node.dependencies.append(dep_internal2)

                        dep_internal.node.dependencies = list()

                    check_existence = any([dep_internal.node == a.node for a in main_node.dependencies])
                    if not check_existence:
                        main_node.dependencies.append(dep_internal)

            out_node.dependencies = list()

        def _get_breakable_output(node, main_node):
            for out_node in node.outputs:
                if out_node.type in BREAKABLE_NODES:
                    if (len(out_node.dependencies) > 0) and (out_node.type in BREAKABLE_NODES):
                        #find a node thats
                        # if node has already some dependencies then we copy them
                        _rec_dependencies_copy(out_node, main_node)
                    else:
                        dep = Dependency(node=out_node, type="out", shift=0)
                        main_node.dependencies.append(dep)

                    if main_node != out_node:
                        # add dependency
                        dep2 = Dependency(node=main_node, type="bro", shift=0)
                        out_node.dependencies=list()
                        out_node.dependencies.append(dep2)
                else:
                    # print("Found next dependent")
                    _get_breakable_output(out_node, main_node)

        BREAKABLE_NODES = (OPTYPE.CONV, OPTYPE.LINEAR, OPTYPE.BN)

        # any input needs to track all inputs...
        for module, node in module_to_node.items():
            # if node.type != OPTYPE.CONV:
            #     continue
            if node.type != OPTYPE.BN:
                continue

            _get_breakable_input(node, node)
            _get_breakable_output(node, node)


    
    def _obtain_forward_graph(self, model, example_inputs, output_transform):
        #PAVLO: basically this is based on torchviz and tracks down the autograd path. Before that we create hooks to remember which modules are used and reuse them later. Creates Nodes to keep track of dependencies.
        #module_to_node = { m: Node( m ) for m in model.modules() if isinstance( m, self.PRUNABLE_MODULES ) }
        model.eval().cpu()
        # Get grad_fn from prunable modules
        grad_fn_to_module = {}

        visited = {}
        def _record_module_grad_fn(module, inputs, outputs):
            if module not in visited:
                visited[module] = 1
            else:
                visited[module] += 1
            grad_fn_to_module[outputs.grad_fn] = module
        
        hooks = [m.register_forward_hook(_record_module_grad_fn) for m in model.modules() if isinstance( m, self.PRUNABLE_MODULES ) ]
        out = model(example_inputs)
        for hook in hooks:
            hook.remove()
        reused = [ m for (m, count) in visited.items() if count>1 ]
        # create nodes and dummy modules
        module_to_node = {}
        def _build_graph(grad_fn):
            module = grad_fn_to_module.get(grad_fn)
            if module is not None and module in module_to_node and module not in reused:
                return module_to_node[module]

            if module is None:
                if not hasattr(grad_fn, 'name'):
                    module = _ElementWiseOp() # skip customized modules
                    if self.verbose:
                        print("[Warning] Unrecognized operation: %s. It will be treated as element-wise op"%( str(grad_fn) ))
                elif 'catbackward' in grad_fn.name().lower(): # concat op
                    module = _ConcatOp()
                elif 'splitbackward' in grad_fn.name().lower():
                    module = _SplitOP()
                else:
                    module = _ElementWiseOp()   # All other ops are treated as element-wise ops
                grad_fn_to_module[ grad_fn ] = module # record grad_fn

            if module not in module_to_node:
                node = Node( module, grad_fn, self._module_to_name.get( module, None ) )
                module_to_node[ module ] = node
            else:
                node = module_to_node[module]

            if hasattr(grad_fn, 'next_functions'):
                for f in grad_fn.next_functions:
                    if f[0] is not None:
                        if hasattr( f[0], 'name' ) and 'accumulategrad' in f[0].name().lower(): # skip leaf variables
                            continue
                        input_node = _build_graph(f[0])
                        node.add_input( input_node )
                        input_node.add_output( node )
            return node
        
        if output_transform is not None:
            out = output_transform(out)
            
        if isinstance(out, (list, tuple) ):
            for o in out:
                _build_graph( o.grad_fn )
        else:
            _build_graph( out.grad_fn )
        return module_to_node
    #
    # def _set_fc_index_transform(self, fc_node: Node):
    #     if fc_node.type != OPTYPE.LINEAR:
    #         return
    #     visited = set()
    #     fc_in_features = fc_node.module.in_features
    #     feature_channels = _get_in_node_out_channels(fc_node.inputs[0])
    #     stride = fc_in_features // feature_channels
    #     if stride>1:
    #         for in_node in fc_node.inputs:
    #             for dep in fc_node.dependencies:
    #                 if dep.broken_node==in_node:
    #                     dep.index_transform = _FlattenIndexTransform( stride=stride, reverse=True )
    #
    #             for dep in in_node.dependencies:
    #                 if dep.broken_node == fc_node:
    #                     dep.index_transform = _FlattenIndexTransform( stride=stride, reverse=False )
    #
    # def _set_concat_index_transform(self, cat_node: Node):
    #     if cat_node.type != OPTYPE.CONCAT:
    #         return
    #
    #     chs = []
    #     for n in cat_node.inputs:
    #         chs.append( _get_in_node_out_channels(n) )
    #
    #     offsets = [0]
    #     for ch in chs:
    #         offsets.append( offsets[-1]+ch )
    #     cat_node.module.offsets = offsets
    #
    #     for i, in_node in enumerate(cat_node.inputs):
    #         for dep in cat_node.dependencies:
    #             if dep.broken_node == in_node:
    #                 dep.index_transform = _ConcatIndexTransform( offset=offsets[i:i+2], reverse=True )
    #
    #         for dep in in_node.dependencies:
    #             if dep.broken_node == cat_node:
    #                 dep.index_transform = _ConcatIndexTransform( offset=offsets[i:i+2], reverse=False )
