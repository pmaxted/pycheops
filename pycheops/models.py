# -*- coding: utf-8 -*-
"""
models
======
Models for use within the celerite framework

"""

from __future__ import (absolute_import, division, print_function,
                                unicode_literals)
import numpy as np
from celerite import Model
from itertools import chain
from collections import OrderedDict

__all__ = ["pModel"]

class pModel(Model):
    """
    A subclass of the celerite abstract class Model with Gaussian priors

    Initial parameter values can either be provided as arguments in the same
    order as ``parameter_names`` or by name as keyword arguments.

    Args:
        bounds (Optional[list or dict]): Bounds can be given for each
            parameter setting their minimum and maximum allowed values.
            This parameter can either be a ``list`` (with length
            ``full_size``) or a ``dict`` with named parameters. Any parameters
            that are omitted from the ``dict`` will be assumed to have no
            bounds. These bounds can be retrieved later using the
            :func:`celerite.Model.get_parameter_bounds` method and, by
            default, they are used in the :func:`celerite.Model.log_prior`
            method.

        priors (Optional[list or dict]): priors can be given for each
            parameter setting the mean and standard deviation of the Gaussian
            prior probability distribution.
            This parameter can either be a ``list`` (with length
            ``full_size``) or a ``dict`` with named parameters. Any parameters
            that are omitted from the ``dict`` will be assumed to have no
            priors (but bounds on the parameter, if specified, are applied).
            These priors can be retrieved later using the
            :func:`celerite.Model.get_parameter_priors` method and, by
            default, they are used in the :func:`celerite.Model.log_prior`
            method.
    """

    parameter_names = tuple()

    def __init__(self, *args, **kwargs):
        self.unfrozen_mask = np.ones(self.full_size, dtype=bool)
        self.dirty = True

        # Deal with bounds
        self.parameter_bounds = []
        bounds = kwargs.pop("bounds", dict())
        try:
            # Try to treat 'bounds' as a dictionary
            for name in self.parameter_names:
                self.parameter_bounds.append(bounds.get(name, (None, None)))
        except AttributeError:
            # 'bounds' isn't a dictionary - it had better be a list
            self.parameter_bounds = list(bounds)
        if len(self.parameter_bounds) != self.full_size:
            raise ValueError("the number of bounds must equal the number of "
                             "parameters")
        if any(len(b) != 2 for b in self.parameter_bounds):
            raise ValueError("the bounds for each parameter must have the "
                             "format: '(min, max)'")

        # Deal with priors
        self.parameter_priors = []
        priors = kwargs.pop("priors", dict())
        try:
            # Try to treat 'priors' as a dictionary
            for name in self.parameter_names:
                self.parameter_priors.append(priors.get(name, (None, None)))
        except AttributeError:
            # 'priors' isn't a dictionary - it had better be a list
            self.parameter_priors = list(priors)
        if len(self.parameter_priors) != self.full_size:
            raise ValueError("the number of priors must equal the number of "
                             "parameters")
        if any(len(b) != 2 for b in self.parameter_priors):
            raise ValueError("the priors for each parameter must have the "
                             "format: '(min, max)'")

        # Parameter values can be specified as arguments or keywords
        if len(args):
            if len(args) != self.full_size:
                raise ValueError("expected {0} arguments but got {1}"
                                 .format(self.full_size, len(args)))
            if len(kwargs):
                raise ValueError("parameters must be fully specified by "
                                 "arguments or keyword arguments, not both")
            self.parameter_vector = args

        else:
            # Loop over the kwargs and set the parameter values
            params = []
            for k in self.parameter_names:
                v = kwargs.pop(k, None)
                if v is None:
                    raise ValueError("missing parameter '{0}'".format(k))
                params.append(v)
            self.parameter_vector = params

            if len(kwargs):
                raise ValueError("unrecognized parameter(s) '{0}'"
                                 .format(list(kwargs.keys())))

        # Check the initial prior value
        quiet = kwargs.get("quiet", False)
        if not quiet and not np.isfinite(self.log_prior()):
             raise ValueError("non-finite log prior value")


    def get_parameter_priors(self, include_frozen=False):
        """
        Get a list of the parameter priors
        Args:
            include_frozen (Optional[bool]): Should the frozen parameters be
                included in the returned value? (default: ``False``)

        """
        if include_frozen:
            return self.parameter_priors
        return list(p
                    for p, f in zip(self.parameter_priors, self.unfrozen_mask)
                    if f)

    def log_prior(self):
        """Compute the log prior probability of the current parameters"""
        for p, b in zip(self.parameter_vector, self.parameter_bounds):
            if b[0] is not None and p < b[0]:
                return -np.inf
            if b[1] is not None and p > b[1]:
                return -np.inf
        lp = 0.0
        for p,g  in zip(self.parameter_vector, self.parameter_priors):
            lp -= 0.5*((p-g[0])/g[1])**2

        return lp

    @property
    def parameter_priors(self):
        return list(chain(*(
            m.parameter_priors for m in self.models.values()
        )))
