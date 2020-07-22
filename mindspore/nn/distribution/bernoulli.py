# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Bernoulli Distribution"""
from mindspore.ops import operations as P
from mindspore.ops import composite as C
from .distribution import Distribution
from ._utils.utils import cast_to_tensor, check_prob
from ...common import dtype as mstype

class Bernoulli(Distribution):
    """
    Bernoulli Distribution.

    Args:
        probs (float, list, numpy.ndarray, Tensor, Parameter): probability of 1 as outcome.
        seed (int): seed to use in sampling. Default: 0.
        dtype (mindspore.dtype): type of the distribution. Default: mstype.int32.
        name (str): name of the distribution. Default: Bernoulli.

    Note:
        probs should be proper probabilities (0 <= p <= 1).
        Dist_spec_args is probs.

    Examples:
        >>> # To initialize a Bernoulli distribution of prob 0.5
        >>> n = nn.Bernoulli(0.5, dtype=mstype.int32)
        >>>
        >>> # The following create two independent Bernoulli distributions
        >>> n = nn.Bernoulli([0.5, 0.5], dtype=mstype.int32)
        >>>
        >>> # A Bernoulli distribution can be initilize without arguments
        >>> # In this case, probs must be passed in through construct.
        >>> n = nn.Bernoulli(dtype=mstype.int32)
        >>>
        >>> # To use Bernoulli distribution in a network
        >>> class net(Cell):
        >>>     def __init__(self):
        >>>         super(net, self).__init__():
        >>>         self.b1 = nn.Bernoulli(0.5, dtype=mstype.int32)
        >>>         self.b2 = nn.Bernoulli(dtype=mstype.int32)
        >>>
        >>>     # All the following calls in construct are valid
        >>>     def construct(self, value, probs_b, probs_a):
        >>>
        >>>         # Similar to calls can be made to other probability functions
        >>>         # by replacing 'prob' with the name of the function
        >>>         ans = self.b1('prob', value)
        >>>         # Evaluate with the respect to distribution b
        >>>         ans = self.b1('prob', value, probs_b)
        >>>
        >>>         # Additional probs must be passed in through construct
        >>>         ans = self.b2('prob', value, probs_a)
        >>>
        >>>         # Functions 'sd', 'var', 'entropy' have the same usage with 'mean'
        >>>         # Will return [0.0]
        >>>         ans = self.b1('mean')
        >>>         # Will return mean_b
        >>>         ans = self.b1('mean', probs_b)
        >>>
        >>>         # Additional probs must be passed in through construct
        >>>         ans = self.b2('mean', probs_a)
        >>>
        >>>         # Usage of 'kl_loss' and 'cross_entropy' are similar
        >>>         ans = self.b1('kl_loss', 'Bernoulli', probs_b)
        >>>         ans = self.b1('kl_loss', 'Bernoulli', probs_b, probs_a)
        >>>
        >>>         # Additional probs must be passed in through construct
        >>>         ans = self.b2('kl_loss', 'Bernoulli', probs_b, probs_a)
        >>>
        >>>         # Sample Usage
        >>>         ans = self.b1('sample')
        >>>         ans = self.b1('sample', (2,3))
        >>>         ans = self.b1('sample', (2,3), probs_b)
        >>>         ans = self.b2('sample', (2,3), probs_a)
    """

    def __init__(self,
                 probs=None,
                 seed=0,
                 dtype=mstype.int32,
                 name="Bernoulli"):
        """
        Constructor of Bernoulli distribution.
        """
        param = dict(locals())
        super(Bernoulli, self).__init__(dtype, name, param)
        if probs is not None:
            self._probs = cast_to_tensor(probs, dtype=mstype.float32)
            check_prob(self.probs)
        else:
            self._probs = probs
        self.seed = seed

        # ops needed for the class
        self.cast = P.Cast()
        self.const = P.ScalarToArray()
        self.dtypeop = P.DType()
        self.erf = P.Erf()
        self.fill = P.Fill()
        self.log = P.Log()
        self.add = P.TensorAdd()
        self.sq = P.Square()
        self.mul = P.Mul()
        self.sqrt = P.Sqrt()
        self.realdiv = P.RealDiv()
        self.shape = P.Shape()
        self.const = P.ScalarToArray()
        self.less = P.Less()
        self.cast = P.Cast()
        self.erf = P.Erf()
        self.select = P.Select()
        self.fill = P.Fill()

    def extend_repr(self):
        if self.is_scalar_batch:
            str_info = f'probs = {self.probs}'
        else:
            str_info = f'batch_shape = {self._broadcast_shape}'
        return str_info

    @property
    def probs(self):
        """
        Returns the probability for the outcome is 1.
        """
        return self._probs

    def _mean(self, name='mean', probs1=None):
        r"""
        .. math::
            MEAN(B) = probs1
        """
        if name == 'mean':
            return self._probs if probs1 is None else probs1
        return None

    def _mode(self, name='mode', probs1=None):
        r"""
        .. math::
            MODE(B) = 1 if probs1 > 0.5 else = 0
        """
        if name == 'mode':
            probs1 = self.probs if probs1 is None else probs1
            prob_type = self.dtypeop(probs1)
            zeros = self.fill(prob_type, self.shape(probs1), 0.0)
            ones = self.fill(prob_type, self.shape(probs1), 1.0)
            comp = self.less(0.5, probs1)
            return self.select(comp, ones, zeros)
        return None

    def _var(self, name='var', probs1=None):
        r"""
        .. math::
            VAR(B) = probs1 * probs0
        """
        if name in self._variance_functions:
            probs1 = self._probs if probs1 is None else probs1
            probs0 = 1.0 - probs1
            return self.mul(probs0, probs1)
        return None

    def _entropy(self, name='entropy', probs=None):
        r"""
        .. math::
            H(B) = -probs0 * \log(probs0) - probs1 * \log(probs1)
        """
        if name == 'entropy':
            probs1 = self._probs if probs is None else probs
            probs0 = 1 - probs1
            return -self.mul(probs0, self.log(probs0)) - self.mul(probs1, self.log(probs1))
        return None

    def _cross_entropy(self, name, dist, probs1_b, probs1_a=None):
        """
        Evaluate cross_entropy between Bernoulli distributions.

        Args:
            name (str): name of the funtion.
            dist (str): type of the distributions. Should be "Bernoulli" in this case.
            probs1_b (Tensor): probs1 of distribution b.
            probs1_a (Tensor): probs1 of distribution a. Default: self._probs.
        """
        if name == 'cross_entropy' and dist == 'Bernoulli':
            return self._entropy(probs=probs1_a) + self._kl_loss(name, dist, probs1_b, probs1_a)
        return None

    def _prob(self, name, value, probs=None):
        r"""
        pmf of Bernoulli distribution.

        Args:
            name (str): name of the function. Should be "prob" when passed in from construct.
            value (Tensor): a Tensor composed of only zeros and ones.
            probs (Tensor): probability of outcome is 1. Default: self._probs.

        .. math::
            pmf(k) = probs1 if k = 1;
            pmf(k) = probs0 if k = 0;
        """
        if name in self._prob_functions:
            probs1 = self._probs if probs is None else probs
            probs0 = 1.0 - probs1
            return (probs1 * value) + (probs0 * (1.0 - value))
        return None

    def _cdf(self, name, value, probs=None):
        r"""
        cdf of Bernoulli distribution.

        Args:
            name (str): name of the function.
            value (Tensor): value to be evaluated.
            probs (Tensor): probability of outcome is 1. Default: self._probs.

        .. math::
            cdf(k) = 0 if k < 0;
            cdf(k) = probs0 if 0 <= k <1;
            cdf(k) = 1 if k >=1;
        """
        if name in self._cdf_survival_functions:
            probs1 = self.probs if probs is None else probs
            prob_type = self.dtypeop(probs1)
            value = value * self.fill(prob_type, self.shape(probs1), 1.0)
            probs0 = 1.0 - probs1 * self.fill(prob_type, self.shape(value), 1.0)
            comp_zero = self.less(value, 0.0)
            comp_one = self.less(value, 1.0)
            zeros = self.fill(prob_type, self.shape(value), 0.0)
            ones = self.fill(prob_type, self.shape(value), 1.0)
            less_than_zero = self.select(comp_zero, zeros, probs0)
            return self.select(comp_one, less_than_zero, ones)
        return None

    def _kl_loss(self, name, dist, probs1_b, probs1_a=None):
        r"""
        Evaluate bernoulli-bernoulli kl divergence, i.e. KL(a||b).

        Args:
            name (str): name of the funtion.
            dist (str): type of the distributions. Should be "Bernoulli" in this case.
            probs1_b (Tensor): probs1 of distribution b.
            probs1_a (Tensor): probs1 of distribution a. Default: self._probs.

        .. math::
            KL(a||b) = probs1_a * \log(\fract{probs1_a}{probs1_b}) +
                       probs0_a * \log(\fract{probs0_a}{probs0_b})
        """
        if name in self._divergence_functions and dist == 'Bernoulli':
            probs1_a = self._probs if probs1_a is None else probs1_a
            probs0_a = 1.0 - probs1_a
            probs0_b = 1.0 - probs1_b
            return self.mul(probs1_a, self.log(probs1_a / probs1_b)) + self.mul(probs0_a, self.log(probs0_a / probs0_b))
        return None

    def _sample(self, name, shape=(), probs=None):
        """
        Sampling.

        Args:
            name (str): name of the function. Should always be 'sample' when passed in from construct.
            shape (tuple): shape of the sample. Default: ().
            probs (Tensor): probs1 of the samples. Default: self._probs.

        Returns:
            Tensor, shape is shape + batch_shape.
        """
        if name == 'sample':
            probs1 = self._probs if probs is None else probs
            batch_shape = self.shape(probs1)
            sample_shape = shape + batch_shape
            mean_zero = self.const(0.0)
            sd_one = self.const(1.0)
            sqrt_two = self.sqrt(self.const(2.0))
            sample_norm = C.normal(sample_shape, mean_zero, sd_one, self.seed)
            sample_uniform = 0.5 * (1 + self.erf(self.realdiv(sample_norm, sqrt_two)))
            sample = self.less(sample_uniform, probs1)
            sample = self.cast(sample, self._dtype)
            return sample
        return None
