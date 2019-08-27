import numpy as np
from collections import defaultdict, deque

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable


class CNN(nn.Module):
    def __init__(self, args):
        super(CNN, self).__init__()

        self.args = args
        self.num_layers = args.child_num_layers
        self.out_filters = args.child_out_filters
        self.num_branches = args.child_num_branches
        self.num_cells = args.child_num_cells
        self.use_aux_heads = args.child_use_aux_heads
        self.fixed_arc = args.fixed_arc
        self.dataset = args.dataset
        if args.dataset == "CIFAR10":
            self.classes = 10
        elif args.dataset == "Sport8":
            self.classes = 8
        elif args.dataset == "MIT67":
            self.classes = 67
        elif args.dataset == "flowers102":
            self.classes = 102
        pool_distance = self.num_layers // 3
        self.pool_layers = [pool_distance, 2*pool_distance+1]

        # TODO: aux head stuff
        if self.use_aux_heads:
            self.aux_head_indices = [self.pool_layers[-1]+1]
        if args.dataset == "CIFAR10":
            self.stem_conv = nn.Sequential(
                nn.Conv2d(3, self.out_filters*3, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(self.out_filters*3),
            )
        elif args.dataset == "Sport8" or args.dataset == "MIT67" or args.dataset == "flowers102":
            self.stem0 = nn.Sequential(
                nn.Conv2d(3, self.out_filters // 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.out_filters // 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.out_filters // 2, self.out_filters, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.out_filters),
            )

            self.stem1 = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Conv2d(self.out_filters, self.out_filters, 3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(self.out_filters),
            )
        self._compile_model()
        self._init_param(self.modules())

    def forward(self, inputs, dag):
        if self.fixed_arc==None:
            self.normal_arc, self.reduce_arc = dag
        else:
            self.normal_arc, self.reduce_arc = self.fixed_arc[0], self.fixed_arc[1]
        logits, aux_logits = self._get_model(inputs)
        return logits, aux_logits

    def _compile_model(self):
        out_filters = self.out_filters
        if self.dataset == "CIFAR10":
            in_filters = [out_filters*3, out_filters*3]
        else:
            in_filters = [out_filters, out_filters]
        self.add_module('layer', nn.ModuleList())
        for layer_id in range(self.num_layers+2):
            self.layer.append(nn.Module())
            if layer_id not in self.pool_layers:
                self._compile_layer(self.layer[layer_id], layer_id, in_filters, out_filters)
            else:
                out_filters *= 2
                self._compile_reduction(self.layer[layer_id], in_filters, out_filters)
                in_filters = [in_filters[-1], out_filters]
                self._compile_layer(self.layer[layer_id], layer_id, in_filters, out_filters)
            in_filters = [in_filters[-1], out_filters]
            if layer_id == 0:
                if self.dataset == "Sport8" or self.dataset == "MIT67" or self.dataset == "flowers102":
                    self._compile_reduction(self.layer[layer_id].calibrate, in_filters, out_filters)

            if self.use_aux_heads and layer_id in self.aux_head_indices:
                self.add_module('aux_head', nn.Sequential(
                    nn.ReLU(),
                    nn.AvgPool2d(5, stride=3, padding=0),
                    nn.Conv2d(out_filters, 128, kernel_size=1, padding=0),
                    nn.BatchNorm2d(128, track_running_stats=False),
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d(1),
                ))
                self.add_module('aux_fc', nn.Sequential(
                    nn.Linear(128,768),
                    nn.BatchNorm1d(768, track_running_stats=False),
                    nn.ReLU(),
                    nn.Linear(768, self.classes),
                ))

        self.add_module('final_fc', nn.Linear(out_filters, self.classes))

    def _compile_layer(self, module, layer_id, in_filters, out_filters):
        self._compile_calibrate(module, in_filters, out_filters)
        module.add_module('cell', nn.ModuleList())
        for cell_id in range(self.num_cells):
            module.cell.append(nn.ModuleList())
            for i in range(2):
                module.cell[cell_id].append(nn.Module())
                self._compile_cell(module.cell[cell_id][i], cell_id, out_filters)

        param = torch.empty(self.num_cells+2, out_filters**2, 1, 1)
        nn.init.kaiming_normal_(param)
        module.register_parameter('final_conv', nn.Parameter(param))
        module.add_module('final_bn', nn.BatchNorm2d(out_filters, track_running_stats=False))

    def _compile_cell(self, module, curr_cell, out_filters):
        #TODO: not needed?

        module.add_module('three', nn.ModuleList())
        self._compile_conv(module.three, curr_cell, 3, out_filters)
        module.add_module('five', nn.ModuleList())
        self._compile_conv(module.five, curr_cell, 5, out_filters)

    def _compile_conv(self, module, curr_cell, filter_size, out_filters, stack_conv=2):
        num_possible_inputs = curr_cell+2
        for i in range(num_possible_inputs):
            module.append(nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(out_filters, out_filters, filter_size, padding=filter_size//2, groups=out_filters, bias=False),
                nn.Conv2d(out_filters, out_filters, 1, bias=False),
                nn.BatchNorm2d(out_filters, track_running_stats=False),
                nn.ReLU(),
                nn.Conv2d(out_filters, out_filters, filter_size, padding=filter_size//2, groups=out_filters, bias=False),
                nn.Conv2d(out_filters, out_filters, 1, bias=False),
                nn.BatchNorm2d(out_filters, track_running_stats=False),
            ))

    def _compile_calibrate(self, module, in_filters, out_filters):
        module.add_module('calibrate', nn.Module())
        # TODO: Not sure
        if in_filters[0] * 2 == in_filters[1]:
            self._compile_reduction(module.calibrate, in_filters, out_filters)
        if in_filters[0] != out_filters:
            module.calibrate.add_module('pool_x', nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_filters[0], out_filters, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(out_filters, track_running_stats=False),
            ))
        if in_filters[1] != out_filters:
            module.calibrate.add_module('pool_y', nn.Sequential(
                nn.ReLU(),
                nn.Conv2d(in_filters[1], out_filters, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm2d(out_filters, track_running_stats=False),
            ))

    def _compile_reduction(self, module, in_filters, out_filters):
        module.add_module('reduction', nn.Module())
        # TODO: path_conv?
        module.reduction.add_module('path1_conv', nn.Sequential(
            nn.AvgPool2d(kernel_size=1, stride=2, padding=0),
            nn.Conv2d(in_filters[0], out_filters//2, kernel_size=1, padding=0, bias=False),
        ))
        module.reduction.add_module('padding', nn.ZeroPad2d((0,1,0,1)))
        module.reduction.add_module('path2_conv', nn.Sequential(
            nn.AvgPool2d(kernel_size=1, stride=2, padding=0),
            nn.Conv2d(in_filters[0], out_filters//2, kernel_size=1, padding=0, bias=False),
        ))
        module.reduction.add_module('bn', nn.BatchNorm2d(out_filters, track_running_stats=False))

    def _get_model(self, inputs):

        aux_logits = None
        # stem conv
        if self.dataset == "CIFAR10":
            x = self.stem_conv(inputs)
            layers = [x, x]
        elif self.dataset == "Sport8" or self.dataset == "MIT67" or self.dataset == "flowers102":
            s0 = self.stem0(inputs)
            s1 = self.stem1(s0)
            layers = [s0, s1]

        out_filters = self.out_filters
        for layer_id in range(self.num_layers + 2):
            if layer_id not in self.pool_layers:
                if self.fixed_arc is None:
                    x = self._enas_layer(layers, self.layer[layer_id], self.normal_arc, out_filters)
            else:
                out_filters *= 2
                if self.fixed_arc is None:
                    x = self._factorized_reduction(x, self.layer[layer_id].reduction)
                    layers = [layers[-1], x]
                    x = self._enas_layer(layers, self.layer[layer_id], self.reduce_arc, out_filters)
            layers = [layers[-1], x]

            if self.use_aux_heads and layer_id in self.aux_head_indices:
                aux = self.aux_head(x).view(x.size(0), -1)
                aux_logits = self.aux_fc(aux)
            '''self.num_aux_vars = 0
            if self.use_aux_heads and layer_id in self.aux_head_indices and is_training:
                cur_ctx = self._get_context(cur_ctx, 'aux_head', [x.size(1), x.size(2)])
                aux_logits = cur_ctx(x)'''

        x = F.dropout2d(F.adaptive_avg_pool2d(F.relu(x), 1), 0.1)
        x = self.final_fc(x.view(x.size(0),-1))
        # TODO: dropout
        #if is_training and self.keep_prob is not None and self.keep_prob < 1.0:
        #    x = F.dropout(x)
        #x = self.fc(x)
        return x, aux_logits

    def _enas_layer(self, prev_layers, module, arc, out_filters):
        layers = [prev_layers[0], prev_layers[1]]
        layers = self._maybe_calibrate_size(layers, module.calibrate, out_filters)
        used = []
        for cell_id in range(self.num_cells):
            # TODO: dim?
            prev_layers = torch.stack(layers)

            x_id = arc[4 * cell_id]
            x_op = arc[4 * cell_id + 1]
            x = prev_layers[x_id, :, :, :, :]
            x = self._enas_cell(x, module.cell[cell_id][0], cell_id, x_id, x_op)
            x_used = torch.zeros(self.num_cells+2).long()
            x_used[x_id] = 1

            y_id = arc[4 * cell_id + 2]
            y_op = arc[4 * cell_id + 3]
            y = prev_layers[y_id, :, :, :, :]
            y = self._enas_cell(y, module.cell[cell_id][1], cell_id, y_id, y_op)
            y_used = torch.zeros(self.num_cells+2).long()
            y_used[y_id] = 1

            out = x + y
            used.extend([x_used, y_used])
            layers.append(out)

        used_ = torch.zeros(used[0].shape).long()
        for i in range(len(used)):
            used_ = used_ + used[i]
        # TODO
        indices = torch.eq(used_, 0).nonzero().long().view(-1)
        num_outs = indices.size(0)
        out = torch.stack(layers)
        out = out[indices]

        inp = prev_layers[0]
        N, C, H, W = inp.shape
        out = out.transpose(0, 1).contiguous().view(N, num_outs*out_filters, H, W)

        out = F.relu(out)
        w = module.final_conv[indices].view(out_filters, out_filters*num_outs, 1, 1)
        out = F.conv2d(out, w)
        out = module.final_bn(out)

        out = out.view(prev_layers[0].shape)
        return out

    def _maybe_calibrate_size(self, layers, module, out_filters):
        hw = [layer.shape[2] for layer in layers]
        c = [layer.shape[1] for layer in layers]

        x = layers[0]
        if hw[0] != hw[1]:
            assert hw[0] == 2 * hw[1]
            x = F.relu(x)
            x = self._factorized_reduction(x, module.reduction)
        elif c[0] != out_filters:
            x = module.pool_x(x)
        y = layers[1]
        if c[1] != out_filters:
            y = module.pool_y(y)
        return [x, y]

    def _factorized_reduction(self, x, module):
        path1 = module.path1_conv(x)

        path2 = module.padding(x)
        path2 = path2[:,:,1:,1:]
        path2 = module.path2_conv(path2)

        final_path = torch.cat([path1, path2], dim=1)
        final_path = module.bn(final_path)
        return final_path

    def _enas_cell(self, x, module, curr_cell, prev_cell, op_id):
        if op_id == 0:
            out = module.three[prev_cell](x)
        elif op_id == 1:
            out = module.five[prev_cell](x)
        elif op_id == 2:
            out = F.avg_pool2d(x, 3, stride=1, padding=1)
        elif op_id == 3:
            out = F.max_pool2d(x, 3, stride=1, padding=1)
        else:
            out = x

        return out

    def reset_parameters(self):
        pass

    def _init_param(self, module, trainable=True, seed=None):
        for mod in module:
            if type(mod) == nn.Conv2d or type(mod) == nn.Linear:
                nn.init.kaiming_normal_(mod.weight)

