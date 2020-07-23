import torch
import torch.nn as nn
from collections import OrderedDict

from .model_hook import ModelHook
from .stat_tree import StatTree, StatNode
from .reporter import report_format


def get_parent_node(root_node, stat_node_name):
	assert isinstance(root_node, StatNode)

	node = root_node
	names = stat_node_name.split('.')
	for i in range(len(names) - 1):
		node_name = '.'.join(names[0:i+1])
		child_index = node.find_child_index(node_name)
		assert child_index != -1
		node = node.children[child_index]
	return node


def convert_leaf_modules_to_stat_tree(leaf_modules):
	assert isinstance(leaf_modules, OrderedDict)

	create_index = 1
	root_node = StatNode(name='root', parent=None)
	for leaf_module_name, leaf_module in leaf_modules.items():
		names = leaf_module_name.split('.')
		for i in range(len(names)):
			create_index += 1
			stat_node_name = '.'.join(names[0:i+1])
			parent_node = get_parent_node(root_node, stat_node_name)
			node = StatNode(name=stat_node_name, parent=parent_node)
			parent_node.add_child(node)
			if i == len(names) - 1:  # leaf module itself
				node.input_shape = leaf_module.input_shape.numpy().tolist()
				node.output_shape = leaf_module.output_shape.numpy().tolist()
				node.parameter_quantity = leaf_module.parameter_quantity.numpy()[0]
				node.inference_memory = leaf_module.inference_memory.numpy()[0]
				node.MAdd = leaf_module.MAdd.numpy()[0]
				node.Flops = leaf_module.Flops.numpy()[0]
				node.ConvFlops = leaf_module.ConvFlops.numpy()[0]
				node.duration = leaf_module.duration.numpy()[0]
				node.MemRead = leaf_module.MemRead.numpy()[0]
				node.MemWrite = leaf_module.MemWrite.numpy()[0]
	return StatTree(root_node)


class ModelStat(object):
	def __init__(self, model, input_size, query_granularity=1, brief_report=False):
		assert isinstance(model, nn.Module)
		assert isinstance(input_size, (tuple, list)) and len(input_size) == 4
		self.model_hook = ModelHook(model, input_size)
		self.leaf_modules = self.model_hook.retrieve_leaf_modules()
		self.stat_tree = convert_leaf_modules_to_stat_tree(self.leaf_modules)
		self._brief_report = brief_report
		
		if 1 <= query_granularity <= self.stat_tree.root_node.depth:
			self._query_granularity = query_granularity
		else:
			self._query_granularity = self.stat_tree.root_node.depth

	def show_report(self):
		collected_nodes = self.stat_tree.get_collected_stat_nodes(self._query_granularity)
		report = report_format(collected_nodes, self._brief_report)
		print(report)

	def unhook_model(self):
		self.model_hook._unhook_model()

	@property
	def query_granularity(self):
		return self._query_granularity

	@query_granularity.setter
	def query_granularity(self, query_granularity):
		if 1 <= query_granularity <= self.stat_tree.root_node.depth:
			self._query_granularity = query_granularity
		else:
			self._query_granularity = self.stat_tree.root_node.depth

	@property
	def brief_report(self):
		return self._brief_report

	@brief_report.setter
	def brief_report(self, brief_report):
		self._brief_report = brief_report


def stat(model, input_size, query_granularity=1, brief_report=False):
	ms = ModelStat(model, input_size, query_granularity, brief_report)
	ms.show_report()
	ms.unhook_model()
