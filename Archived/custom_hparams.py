# custom_hparams.py
# author: Diego Magdaleno
# Self implementation of HParams class from
# tensorflow.contrib.training.HParams since it was removed from the
# official Tensorflow 2.0.
# Source: http://tensorflow.biotecan.com/python/Python_1.8/tensorflow.
# google.cn/api_docs/python/tf/contrib/training/HParams.html
# https://github.com/tensorflow/tensorflow/blob/r1.8/tensorflow/
# contrib/training/python/training/hparam.py
# Tensorflow 2.4
# Python 3.7
# Windows/MacOS/Linux


import six
import json
import re
import numbers


PARAM_RE = re.compile(r"""
	(?P<name>[a-zA-Z][\w]*)		# variable name: "var" or "x"
	(\[\s*(?P<index>\d+)\s*\])?	# (optional) index: "1" or None
	\s*=\s*
	((?P<val>[^,\[]*)			# single value: "a" or None
	|
	\[(?P<vals>[^\]]*)\])		# list of values: None or "1,2,3"
	($|,\s*)""", re.VERBOSE)


def _parse_fail(name, var_type, value, values):
	raise ValueError(
		"Could not parse hparam '%s' of type '%s' with value '%s' in %s" %
		(name, var_type.__name__, value, values)
	)


def _reuse_fail(name, values):
	raise ValueError("Multiple assignments to variable '%s' in %s" % 
		(name, values))


def _process_scalar_value(name, parse_fn, var_type, m_dict, values, results_dictionary):
	try:
		parsed_value = parse_fn(m_dict["val"])
	except ValueError:
		_parse_fail(name, var_type, m_dict["val"], values)

	if not m_dict["index"]:
		if name in results_dictionary:
			_reuse_fail(name, values)
		results_dictionary[name] = parsed_value
	else:
		if name in results_dictionary:
			if not isinstance(results_dictionary.get(name), dict):
				_reuse_fail(name, values)
		else:
			results_dictionary[name] = {}

		index = int(m_dict["index"])

		if index in results_dictionary[name]:
			_reuse_fail("{}[{}]".format(name, index), values)
		results_dictionary[name][index] = parsed_value


def _process_list_value(name, parse_fn, var_type, m_dict, values, results_dictionary):
	if m_dict["index"] is not None:
		raise ValueError("Assignment of a list to a list index.")
	elements = filter(None, re.split('[ ,]', m_dict["vals"]))

	if name in results_dictionary:
		raise _reuse_fail(name, values)
	try:
		results_dictionary[name] = [parse_fn(e) for e in elements]
	except ValueError:
		_parse_fail(name, var_type, m_dict["vals"], values)


def parse_values(values, type_map):
	results_dictionary = {}
	pos = 0
	while pos < len(values):
		m = PARAM_RE.match(values, pos)
		if not m:
			raise ValueError("Malformed hyperparameter value: %s" % values[pos:])

		pos = m.end()

		m_dict = m.groupdict()
		name = m_dict["name"]
		if name not in type_map:
			raise ValueError("Unknown hyperparameter type for %s" % name)
		type_ = type_map[name]

		if type_ == bool:
			def parse_bool(value):
				if value in ["true", "True"]:
					return True
				elif value in ["false", "False"]:
					return False
				else:
					try:
						return bool(int(value))
					except ValueError:
						_parse_fail(name, type_, value, values)
			parse = parse_bool
		else:
			parse = type_

		if m_dict["val"] is not None:
			_process_scalar_value(name, parse, type_, m_dict, values, results_dictionary)
		elif m_dict["vals"] is not None:
			_process_list_value(name, parse, type_, m_dict, values, results_dictionary)
		else:
			_parse_fail(name, type_, "", values)

		return results_dictionary


def _cast_to_type_if_compatible(name, param_type, value):
	fail_msg = (
		"Could not cast hparam '%s' of type '%s' from value '%r'" %
		(name, param_type, value)
	)

	if issubclass(param_type, type(None)):
		return value

	if issubclass(param_type, (six.string_types, six.binary_types)) and\
			not isinstance(value, (six.string_types, six.binary_types)):
		raise ValueError(fail_msg)

	if issubclass(param_type, bool) != isinstance(value, bool):
		raise ValueError(fail_msg)

	if issubclass(param_type, numbers.Integral) and\
			not isinstance(value. numbers.Integral):
		raise ValueError(fail_msg)

	if issubclass(param_type, numbers.Number) and\
			not isinstance(value. numbers.Number):
		raise ValueError(fail_msg)

	return param_type(value)


class HParams:
	_HAS_DYNAMIC_ATTRIBUTES = True

	def __init__(self, hparam_def=None, model_structure=None, **kwargs):
		self._hparam_types = {}
		self._model_structure = model_structure

		if hparam_def:
			#self._init_from_proto(hparam_def)
			if kwargs:
				raise ValueError("hparam_def and initialization "
					"values are mutually exclusive")
			raise Exception("hparam_def is currently undefined. "
					"Please use kwargs to pass in all HParam "
					"attributes.")
		else:
			for name, value in six.iteritems(kwargs):
				self.add_hparam(name, value)


	def __contains__(self, key):
		return key in self._hparam_types


	# Add {name, value} pair to hyperparameters.
	# @param: name, name of the hyperparameter.
	# @param: value, value of the hyperparameter. Can be one of the
	#	following types: int, float, str, int list, float list, or str
	#	list.
	# @raises: ValueError, if one of the arguments is invalid.
	# @return: returns nothing
	def add_hparam(self, name, value):
		if getattr(self, name, None) is not None:
			raise ValueError("Hyperparameter name is reserved: %s" % name)
		if isinstance(value, (list, tuple)):
			if not value:
				raise ValueError("Multi-valued hyperparameters cannot be empty: %s" % name)
			self._hparam_types[name] = (type(value[0]), True)
		else:
			self._hparam_types[name] = (type(value), True)
		setattr(self, name, value)


	@staticmethod
	def from_proto(self, hparam_def, import_scope=None):
		pass


	# Return the value of key if it exists, else default.
	# @param: key, 
	# @param: default, 
	# @return: return the value of key if it exists, else default.
	def get(self, key, default=None):
		if key in self._hparam_types:
			if default is not None:
				param_type, is_param_list = self._hparam_types[key]
				type_str = "list<%s>" % param_type if is_param_list else str(param_type)
				fail_msg = (
					"HParam '%s' of type '%s' is incompatible with default=%s" %
					(key, type_str, default)
				)

				is_default_list = isinstance(default, list)
				if is_param_list != is_default_list:
					raise ValueError(fail_msg)

				try:
					if is_default_list:
						for value in default:
							_cast_to_type_if_compatible(key, param_type, value)
					else:
						_cast_to_type_if_compatible(key, param_type, default)
				except ValueError as e:
					raise ValueError("%s. %s" % (fail_msg, e))

			return getattr(self, key)

		return default


	def get_model_structure(self):
		return self._model_structure


	def override_from_dict(self, values_dict):
		for name, value in values_dict.items():
			self.set_hparam(name, value)
		return self


	def parse(self, values):
		type_map = dict()
		for name, t in self._hparam_types.items():
			param_type, _ = t
			type_map[name] = param_type

		values_map = parse_values(values, type_map)
		return self.override_from_dict(values_map)


	def parse_json(self, values_json):
		pass


	def set_from_map(self, values_map):
		pass


	def set_hparam(self, name, value):
		param_type, is_list = self._hparam_types[name]
		if isinstance(value, list):
			if not is_list:
				raise ValueError(
					"Must not pass a list for single-valued parameter: %s" % name
				)
			setattr(self, name, 
				[_cast_to_type_if_compatible(name, param_type, v) for v in value]
			)
		else:
			if is_list:
				raise ValueError(
					"Must pass a list for a multi-valued parameter: %s" % name
				)
			setattr(self, name, _cast_to_type_if_compatible(name, param_type, value))


	def set_model_structure(self, model_structure):
		self._model_structure = model_structure


	def to_json(self, indent=None, separators=None, sort_keys=False):
		return json.dumps(self.values(), indent=indent, separators=separators, sort_keys=sort_keys)


	def to_proto(self, export_scope=None):
		pass


	def values(self):
		return {n: getattr(self, n) for n in self._hparam_types.keys()}