import torch
import torch.nn as nn

from abc import abstractmethod


def _mul_broadcast_shape(*shapes, error_msg=None):
    """
    Compute dimension suggested by multiple tensor indices (supports
    broadcasting).
    """

    # Pad each shape so they have the same number of dimensions.
    num_dims = max(len(shape) for shape in shapes)
    shapes = tuple([1] * (num_dims - len(shape)) + list(shape)
                   for shape in shapes)

    # Make sure that each dimension agrees in size.
    final_size = []
    for size_by_dim in zip(*shapes):
        non_singleton_sizes = tuple(size for size in size_by_dim if size != 1)
        if len(non_singleton_sizes):
            if any(size != non_singleton_sizes[0]
                   for size in non_singleton_sizes):
                if error_msg is None:
                    raise RuntimeError("Shapes are not broadcastable for mul "
                                       "operation")
                else:
                    raise RuntimeError(error_msg)
            final_size.append(non_singleton_sizes[0])
        # In this case - all dimensions are singleton sizes.
        else:
            final_size.append(1)

    return torch.Size(final_size)


def default_postprocess_script(x):
    return x


def postprocess_rbf(dist_mat):
    return dist_mat.div_(-2).exp_()


class Distance(torch.nn.Module):
    def __init__(self, postprocess_script=default_postprocess_script):
        super().__init__()
        self._postprocess = postprocess_script

    def _sq_dist(self, x1, x2, postprocess, x1_eq_x2=False):
        adjustment = x1.mean(-2, keepdim=True)
        x1 = x1 - adjustment
        x2 = x2 - adjustment

        # Compute squared distance matrix using quadratic expansion.
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
        x1_pad = torch.ones_like(x1_norm)
        if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
            x2_norm, x2_pad = x1_norm, x1_pad
        else:
            x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
            x2_pad = torch.ones_like(x2_norm)

        x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
        res = x1_.matmul(x2_.transpose(-2, -1))

        if x1_eq_x2 and not x1.requires_grad and not x2.requires_grad:
            res.diagonal(dim1=-2, dim2=-1).fill_(0)

        # Zero out negative values.
        res.clamp_min_(0)
        return self._postprocess(res) if postprocess else res

    def _dist(self, x1, x2, postprocess, x1_eq_x2=False):
        res = self._sq_dist(x1, x2, postprocess=False, x1_eq_x2=x1_eq_x2)
        res = res.clamp_min_(1e-30).sqrt_()
        return self._postprocess(res) if postprocess else res


class Kernel(nn.Module):
    """
    Base class for GP kernels, all of which are assumed to include a scale
    and lengthscale, as well as the option for ARD lengthscales.
    """
    def __init__(self, ard_num_dims=None, batch_shape=torch.Size([]),
                 lengthscale=1., outputscale=1., train_hypers=False):
        super().__init__()

        self.ard_num_dims = ard_num_dims
        self._batch_shape = batch_shape
        self.train_hypers = train_hypers
        outputscale = torch.ones(*self._batch_shape) * outputscale if len(
            self._batch_shape) else torch.tensor(outputscale)

        if self.train_hypers:
            self.register_parameter(
                "log_outputscale",
                nn.Parameter(outputscale.log(), requires_grad=True))
        else:
            self.register_buffer("log_outputscale", outputscale.log())

        lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims
        lengthscale = torch.ones(
            *self._batch_shape, 1, lengthscale_num_dims) * lengthscale

        if self.train_hypers:
            self.register_parameter(
                "log_lengthscale",
                nn.Parameter(lengthscale.log(), requires_grad=True))
        else:
            self.register_buffer("log_lengthscale", lengthscale.log())

        self.distance_module = None

    @property
    def batch_shape(self):
        return self._batch_shape

    @batch_shape.setter
    def batch_shape(self, val):
        self._batch_shape = val

    @abstractmethod
    def forward(self, x1, x2, diag=False):
        raise NotImplementedError

    @property
    def lengthscale(self):
        return self.log_lengthscale.exp()

    @lengthscale.setter
    def lengthscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)

        if self.train_hypers:
            self.log_lengthscale.data = value.log()
        else:
            self.log_lengthscale = value.log()

    @property
    def outputscale(self):
        return self.log_outputscale.exp()

    @outputscale.setter
    def outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)

        if self.train_hypers:
            self.log_outputscale.data = value.log()
        else:
            self.log_outputscale = value.log()

    def covar_dist(self, x1, x2, diag=False, square_dist=False,
                   dist_postprocess_func=default_postprocess_script,
                   postprocess=True):
        x1_eq_x2 = torch.equal(x1, x2)

        # Torch scripts expect tensors.
        postprocess = torch.tensor(postprocess)

        # Cache the Distance object or else JIT will recompile every time.
        if not self.distance_module or self.distance_module._postprocess != \
                dist_postprocess_func:
            self.distance_module = Distance(dist_postprocess_func)

        if diag:
            # Special case the diagonal because we can return all zeros most
            # of the time.
            if x1_eq_x2:
                res = torch.zeros(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype,
                                  device=x1.device)
                if postprocess:
                    res = dist_postprocess_func(res)
                return res
            else:
                res = torch.norm(x1 - x2, p=2, dim=-1)
                if square_dist:
                    res = res.pow(2)
            if postprocess:
                res = dist_postprocess_func(res)
            return res

        elif square_dist:
            res = self.distance_module._sq_dist(x1, x2, postprocess, x1_eq_x2)
        else:
            res = self.distance_module._dist(x1, x2, postprocess, x1_eq_x2)

        return res

    def __call__(self, x1, x2=None, diag=False):
        x1_, x2_ = x1, x2

        # Give x1_ and x2_ a last dimension, if necessary.
        if x1_.ndimension() == 1:
            x1_ = x1_.unsqueeze(1)
        if x2_ is not None:
            if x2_.ndimension() == 1:
                x2_ = x2_.unsqueeze(1)
            if not x1_.size(-1) == x2_.size(-1):
                raise RuntimeError(
                    "x1_ and x2_ must have the same number of dimensions!")

        if x2_ is None:
            x2_ = x1_

        # Check that ard_num_dims matches the supplied number of dimensions.
        if self.ard_num_dims is not None and self.ard_num_dims != x1_.size(-1):
            raise RuntimeError("Expected the input to have {} dimensionality "
                               "(based on the ard_num_dims argument). Got "
                               "{}.".format(self.ard_num_dims, x1_.size(-1)))

        if diag:
            res = self.forward(x1_, x2_, diag=True)
            # Did this Kernel eat the diag option?
            # We can call diag on the output.
            if res.dim() == x1_.dim() and res.shape[-2:] == torch.Size(
                    (x1_.size(-2), x2_.size(-2))):
                res = res.diag()
            return res

        else:
            res = self.forward(x1_, x2_)
            return res


class RBFKernel(Kernel):

    def forward(self, x1, x2, diag=False):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        covar_dist = self.covar_dist(
            x1_, x2_, square_dist=True, diag=diag,
            dist_postprocess_func=postprocess_rbf, postprocess=True)

        outputscales = self.outputscale
        if diag:
            outputscales = outputscales.unsqueeze(-1)
            return covar_dist * outputscales
        else:
            outputscales = outputscales.view(*outputscales.shape, 1, 1)
            return covar_dist.mul(outputscales)
