import torch
import time
import numpy as np
from darts.utils import concat

class Architecture(object):
    """Architecture construct by our methods

    Attributes:
        model: The nn.Module indicate current architecture
        args:
            momentum: float, momentum
            weight_decay: float, weight decay
            arch_learning_rate: float, learning rate for architecture alpha
            model: torch.nn.Module, model

    """

    def __init__(self, model, args):
        self.network_momentum = args.momentum
        self.network_weight_decay = args.weight_decay
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.arch_parameters(),
                         lr=args.arch_learning_rate, betas=(0.5, 0.999),
                         weight_decay=args.arch_weight_decay)

    def _update_model_parameters(self, input, target, eta, network_optimizer):
        """Update unrolled model parameter by training data

        Args:
            input: tensor, input data
            target: tensor, target data for input
            eta:
            network_optimizer:

        Returns:

        """

        loss = self.model._loss(input, target)
        theta = concat(self.model.parameters()).data

        try:
            moment = concat(network_optimizer.state[v]['momentum_buffer']
                                  for v in self.model.parameters()).mul_(self.network_momentum)
        except:
            moment = torch.zeros_like(theta)

        dtheta = concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta

        # w' = w - eta(moment+dw) = theta.sub(eta, moment+dtheta)
        unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))

        return unrolled_model

    def step(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer, unrolled):
        """Do one step of gradient descent for architecture parameters

        Args:
            input_train: A tensor with N * C * H * W for training data
            target_train: A tensor with N * 1 for training target
            input_valid: A tensor with N * C * H * W for validation data
            target_valid: A tensor with N * 1 for validation target
            eta:
            network_optimizer:
            unrolled: A boolean indicate whether to use 2-order approx or not
        """

        self.optimizer.zero_grad()
        # Compute the gradient of architecture's param by backprop
        if unrolled:
            # use 2-order approximation
            # we do forward times: 4; backward times: 4
            self._backward_step_unrolled(input_train, target_train, input_valid, target_valid, eta, network_optimizer)
        else:
            # use 1-order
            # we do forward times: 2; backward times: 2
            self._backward_step(input_valid, target_valid)

        self.optimizer.step()

    def _backward_step(self, input_valid, target_valid):
        loss = self.model._loss(input_valid, target_valid)
        loss.backward()

    def _backward_step_unrolled(self, input_train, target_train, input_valid, target_valid, eta, network_optimizer):
        """Backward step by 2-order approximation for architecture parameter alpha

        :param input_train:
        :param target_train:
        :param input_valid:
        :param target_valid:
        :param eta: A scalar indicate learning rate for model parameters
        :param network_optimizer:
        :return:
        """
        model_updated = self._update_model_parameters(input_train, target_train, eta, network_optimizer)

        model_updated_loss = model_updated._loss(input_valid, target_valid)

        model_updated_loss.backward()
        dalpha = [v.grad for v in model_updated.arch_parameters()]
        vector = [v.grad.data for v in model_updated.parameters()]
        implicit_grads = self._second_order_approximation(vector, input_train, target_train)

        for g, ig in zip(dalpha, implicit_grads):
            g.data.sub_(eta, ig.data)

        # Update the model architecture, finally!
        for v, g in zip(self.model.arch_parameters(), dalpha):
            if v.grad is None:
                v.grad = torch.tensor(g.data)
            else:
                v.grad.data.copy_(g.data)


    def _construct_model_from_theta(self, theta):
        """Construct a new model from parameters: theta

        Args:
            theta: A array indicate the update weight for current model

        Returns:

        """

        model_new = self.model.new()
        model_dict = self.model.state_dict()

        params, offset = {}, 0
        for k, v in self.model.named_parameters():
            v_length = np.prod(v.size())
            params[k] = theta[offset: offset+v_length].view(v.size())
            offset += v_length

        assert offset == len(theta)
        # overwrite entries in the existing state dict
        model_dict.update(params)
        # Todo(zbabby:2018/10/6) load state_dict need about 3 second in pytorch 4.x later, odd!
        # Todo(zbabby:2018/10/6) Because the strict default is True, and pytorch0.4.x later will cost too much time for checking the keys!!!
        # Since we can guarantee the keys are match ,so strict == False
        model_new.load_state_dict(model_dict, strict=False)

        return model_new.cuda() # copy into gpu

    def _second_order_approximation(self, vector, input, target, r=1e-2):
        """Second order approximation for compute the architecture parameter gradient dalpha

        Args:
            vector: A tensor object indicate the gradient of validation Loss with updated w: w'
            input:  A tensor object with N * C * H * W of training data
            target: A tensor object with N * 1 of target data
            r:      A constant scale

        Returns:

        """
        epsilon = r / concat(vector).norm()
        for p, v in zip(self.model.parameters(), vector):
            p.data.add_(epsilon, v) # w + epsilon*DL_w_train(w)
        loss = self.model._loss(input, target)
        # dL_alpha_train(w')
        grads_positive = torch.autograd.grad(loss, self.model.arch_parameters())

        for p, v in zip(self.model.parameters(), vector):
            p.data.sub_(epsilon, v)
        loss = self.model._loss(input, target)
        grads_negative = torch.autograd.grad(loss, self.model.arch_parameters())

        # # restore the old state
        # for p, v in zip(self.model.parameters(), vector):
        #     p.data.add_(epsilon, v)

        return [(x-y).div_(2*epsilon) for x, y in zip(grads_positive, grads_negative)]








