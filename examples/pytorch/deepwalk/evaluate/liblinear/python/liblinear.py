#!/usr/bin/env python

from ctypes import *
from ctypes.util import find_library
from os import path
import sys

try:
	dirname = path.dirname(path.abspath(__file__))
	if sys.platform == 'win32':
		liblinear = CDLL(path.join(dirname, r'..\windows\liblinear.dll'))
	else:
		liblinear = CDLL(path.join(dirname, '../liblinear.so.1'))
except:
# For unix the prefix 'lib' is not considered.
	if find_library('linear'):
		liblinear = CDLL(find_library('linear'))
	elif find_library('liblinear'):
		liblinear = CDLL(find_library('liblinear'))
	else:
		raise Exception('LIBLINEAR library not found.')

# Construct constants
SOLVER_TYPE = ['L2R_LR', 'L2R_L2LOSS_SVC_DUAL', 'L2R_L2LOSS_SVC', 'L2R_L1LOSS_SVC_DUAL',\
		'MCSVM_CS', 'L1R_L2LOSS_SVC', 'L1R_LR', 'L2R_LR_DUAL', \
		None, None, None, \
		'L2R_L2LOSS_SVR', 'L2R_L2LOSS_SVR_DUAL', 'L2R_L1LOSS_SVR_DUAL']
for i, s in enumerate(SOLVER_TYPE): 
	if s is not None: exec("%s = %d" % (s , i))

PRINT_STRING_FUN = CFUNCTYPE(None, c_char_p)
def print_null(s): 
	return 

def genFields(names, types): 
	return list(zip(names, types))

def fillprototype(f, restype, argtypes): 
	f.restype = restype
	f.argtypes = argtypes

class feature_node(Structure):
	_names = ["index", "value"]
	_types = [c_int, c_double]
	_fields_ = genFields(_names, _types)

	def __str__(self):
		return '%d:%g' % (self.index, self.value)

def gen_feature_nodearray(xi, feature_max=None, issparse=True):
	if isinstance(xi, dict):
		index_range = xi.keys()
	elif isinstance(xi, (list, tuple)):
		xi = [0] + xi  # idx should start from 1
		index_range = range(1, len(xi))
	else:
		raise TypeError('xi should be a dictionary, list or tuple')

	if feature_max:
		assert(isinstance(feature_max, int))
		index_range = filter(lambda j: j <= feature_max, index_range)
	if issparse: 
		index_range = filter(lambda j:xi[j] != 0, index_range)

	index_range = sorted(index_range)
	ret = (feature_node * (len(index_range)+2))()
	ret[-1].index = -1 # for bias term
	ret[-2].index = -1
	for idx, j in enumerate(index_range):
		ret[idx].index = j
		ret[idx].value = xi[j]
	max_idx = 0
	if index_range : 
		max_idx = index_range[-1]
	return ret, max_idx

class problem(Structure):
	_names = ["l", "n", "y", "x", "bias"]
	_types = [c_int, c_int, POINTER(c_double), POINTER(POINTER(feature_node)), c_double]
	_fields_ = genFields(_names, _types)

	def __init__(self, y, x, bias = -1):
		if len(y) != len(x) :
			raise ValueError("len(y) != len(x)")
		self.l = l = len(y)
		self.bias = -1

		max_idx = 0
		x_space = self.x_space = []
		for i, xi in enumerate(x):
			tmp_xi, tmp_idx = gen_feature_nodearray(xi)
			x_space += [tmp_xi]
			max_idx = max(max_idx, tmp_idx)
		self.n = max_idx

		self.y = (c_double * l)()
		for i, yi in enumerate(y): self.y[i] = y[i]

		self.x = (POINTER(feature_node) * l)() 
		for i, xi in enumerate(self.x_space): self.x[i] = xi

		self.set_bias(bias)

	def set_bias(self, bias):
		if self.bias == bias:
			return 
		if bias >= 0 and self.bias < 0: 
			self.n += 1
			node = feature_node(self.n, bias)
		if bias < 0 and self.bias >= 0: 
			self.n -= 1
			node = feature_node(-1, bias)

		for xi in self.x_space:
			xi[-2] = node
		self.bias = bias


class parameter(Structure):
	_names = ["solver_type", "eps", "C", "nr_weight", "weight_label", "weight", "p"]
	_types = [c_int, c_double, c_double, c_int, POINTER(c_int), POINTER(c_double), c_double]
	_fields_ = genFields(_names, _types)

	def __init__(self, options = None):
		if options == None:
			options = ''
		self.parse_options(options)

	def __str__(self):
		s = ''
		attrs = parameter._names + list(self.__dict__.keys())
		values = map(lambda attr: getattr(self, attr), attrs) 
		for attr, val in zip(attrs, values):
			s += (' %s: %s\n' % (attr, val))
		s = s.strip()

		return s

	def set_to_default_values(self):
		self.solver_type = L2R_L2LOSS_SVC_DUAL
		self.eps = float('inf')
		self.C = 1
		self.p = 0.1
		self.nr_weight = 0
		self.weight_label = (c_int * 0)()
		self.weight = (c_double * 0)()
		self.bias = -1
		self.cross_validation = False
		self.nr_fold = 0
		self.print_func = cast(None, PRINT_STRING_FUN)

	def parse_options(self, options):
		if isinstance(options, list):
			argv = options
		elif isinstance(options, str):
			argv = options.split()
		else:
			raise TypeError("arg 1 should be a list or a str.")
		self.set_to_default_values()
		self.print_func = cast(None, PRINT_STRING_FUN)
		weight_label = []
		weight = []

		i = 0
		while i < len(argv) :
			if argv[i] == "-s":
				i = i + 1
				self.solver_type = int(argv[i])
			elif argv[i] == "-c":
				i = i + 1
				self.C = float(argv[i])
			elif argv[i] == "-p":
				i = i + 1
				self.p = float(argv[i])
			elif argv[i] == "-e":
				i = i + 1
				self.eps = float(argv[i])
			elif argv[i] == "-B":
				i = i + 1
				self.bias = float(argv[i])
			elif argv[i] == "-v":
				i = i + 1
				self.cross_validation = 1
				self.nr_fold = int(argv[i])
				if self.nr_fold < 2 :
					raise ValueError("n-fold cross validation: n must >= 2")
			elif argv[i].startswith("-w"):
				i = i + 1
				self.nr_weight += 1
				nr_weight = self.nr_weight
				weight_label += [int(argv[i-1][2:])]
				weight += [float(argv[i])]
			elif argv[i] == "-q":
				self.print_func = PRINT_STRING_FUN(print_null)
			else :
				raise ValueError("Wrong options")
			i += 1

		liblinear.set_print_string_function(self.print_func)
		self.weight_label = (c_int*self.nr_weight)()
		self.weight = (c_double*self.nr_weight)()
		for i in range(self.nr_weight): 
			self.weight[i] = weight[i]
			self.weight_label[i] = weight_label[i]

		if self.eps == float('inf'):
			if self.solver_type in [L2R_LR, L2R_L2LOSS_SVC]:
				self.eps = 0.01
			elif self.solver_type in [L2R_L2LOSS_SVR]:
				self.eps = 0.001
			elif self.solver_type in [L2R_L2LOSS_SVC_DUAL, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L2R_LR_DUAL]:
				self.eps = 0.1
			elif self.solver_type in [L1R_L2LOSS_SVC, L1R_LR]:
				self.eps = 0.01
			elif self.solver_type in [L2R_L2LOSS_SVR_DUAL, L2R_L1LOSS_SVR_DUAL]:
				self.eps = 0.1

class model(Structure):
	_names = ["param", "nr_class", "nr_feature", "w", "label", "bias"]
	_types = [parameter, c_int, c_int, POINTER(c_double), POINTER(c_int), c_double]
	_fields_ = genFields(_names, _types)

	def __init__(self):
		self.__createfrom__ = 'python'

	def __del__(self):
		# free memory created by C to avoid memory leak
		if hasattr(self, '__createfrom__') and self.__createfrom__ == 'C':
			liblinear.free_and_destroy_model(pointer(self))

	def get_nr_feature(self):
		return liblinear.get_nr_feature(self)

	def get_nr_class(self):
		return liblinear.get_nr_class(self)

	def get_labels(self):
		nr_class = self.get_nr_class()
		labels = (c_int * nr_class)()
		liblinear.get_labels(self, labels)
		return labels[:nr_class]

	def is_probability_model(self):
		return (liblinear.check_probability_model(self) == 1)

def toPyModel(model_ptr):
	"""
	toPyModel(model_ptr) -> model

	Convert a ctypes POINTER(model) to a Python model
	"""
	if bool(model_ptr) == False:
		raise ValueError("Null pointer")
	m = model_ptr.contents
	m.__createfrom__ = 'C'
	return m

fillprototype(liblinear.train, POINTER(model), [POINTER(problem), POINTER(parameter)])
fillprototype(liblinear.cross_validation, None, [POINTER(problem), POINTER(parameter), c_int, POINTER(c_double)])

fillprototype(liblinear.predict_values, c_double, [POINTER(model), POINTER(feature_node), POINTER(c_double)])
fillprototype(liblinear.predict, c_double, [POINTER(model), POINTER(feature_node)])
fillprototype(liblinear.predict_probability, c_double, [POINTER(model), POINTER(feature_node), POINTER(c_double)])

fillprototype(liblinear.save_model, c_int, [c_char_p, POINTER(model)])
fillprototype(liblinear.load_model, POINTER(model), [c_char_p])

fillprototype(liblinear.get_nr_feature, c_int, [POINTER(model)])
fillprototype(liblinear.get_nr_class, c_int, [POINTER(model)])
fillprototype(liblinear.get_labels, None, [POINTER(model), POINTER(c_int)])

fillprototype(liblinear.free_model_content, None, [POINTER(model)])
fillprototype(liblinear.free_and_destroy_model, None, [POINTER(POINTER(model))])
fillprototype(liblinear.destroy_param, None, [POINTER(parameter)])
fillprototype(liblinear.check_parameter, c_char_p, [POINTER(problem), POINTER(parameter)])
fillprototype(liblinear.check_probability_model, c_int, [POINTER(model)])
fillprototype(liblinear.set_print_string_function, None, [CFUNCTYPE(None, c_char_p)])
