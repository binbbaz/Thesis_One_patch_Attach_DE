from typing import Tuple, Union, Sequence

import numpy as np
import torch
from scipy.optimize import differential_evolution
from torchattacks.attack import Attack
import torch.nn.functional as F
import matplotlib.pyplot as plt

DimensionTupleType = Tuple[Union[int, float], Union[int, float]]
DimensionType = Union[DimensionTupleType, Sequence[DimensionTupleType]]


class OnePatchAttack(Attack):
    def __init__(self, model,
                 population: int = 1,
                 mutation_rate: float = (0.5, 1),
                 crossover_rate: float = 0.8,
                 n_patches: int = 4,
                 max_iterations: int = 1000
        ):

        super().__init__("OnePatchAttack", model)

        if n_patches <= 0:
            raise ValueError('n_patches must be >0'
                             '({})'.format(n_patches))

        self.alpha = alpha
        self.beta = beta
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = population
        self.max_iterations = max_iterations
        self.n_patches = n_patches

        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):

        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        bs, c, h, w = images.shape

        bounds = []
        for _ in range(self.n_patches):
            bounds.extend([(0, w), (0, h), (0, 1), (0, 1), (0, 1)])

        iteration_stats = []
        adv_images = []

        for idx in range(bs):
            image, label = images[idx:idx + 1], labels[idx:idx + 1]
            source_label = label

            if self.get_mode() != 'default':
                label = self._get_target_label(image, label)

            f, c = self._get_fun(image, label,
                                 target_attack=self._targeted,
                                 source_label=source_label)

            solution = differential_evolution(func=f,
                                              callback=c,
                                              bounds=bounds,
                                              maxiter=self.max_iterations,
                                              popsize=self.population,
                                              mutation=self.mutation_rate,
                                              # init='random',
                                              recombination=self.crossover_rate,
                                              atol=-1,
                                              disp=False,
                                              polish=False)

            iteration_stats.append(solution.nfev)
            adv_image = self._perturb(image, solution.x)
            #print("in forward")
            adv_images.append(adv_image)

        self.required_iterations = iteration_stats
        adv_images = torch.cat(adv_images)
        return adv_images

    def _get_prob(self, image):
        out = self.model(image.to(self.device))
        prob = F.softmax(out, dim=1)
        return prob.detach().cpu().numpy()

    def _get_fun(self, img, label, target_attack=False, source_label=None):
        img = img.to(self.device)

        if isinstance(label, torch.Tensor):
            label = label.cpu().numpy()

        @torch.no_grad()
        def func(solution, solution_as_perturbed=False):
            pert_image = self._perturb(img, solution)
            #print("in func")
            p = self._get_prob(pert_image)
            p = p[np.arange(len(p)), label]

            if target_attack:
                p = 1 - p

            return p

        @torch.no_grad()
        def callback(solution, convergence, solution_as_perturbed=False):
            pert_image = self._perturb(img, solution)
            #print("in callback")
            p = self._get_prob(pert_image)[0]
            mx = np.argmax(p)

            if target_attack:
                return mx == label
            else:
                return mx != label

        return func, callback

    def _perturb(self, img, solution):
        pl = 5
        for i in range(self.n_patches):
            patch = solution[i * pl: (i + 1) * pl]
            x, y , r ,g, b = patch
            p_h, p_w = 16, 16

            imgs = img.cpu().detach().numpy()
            imgs = imgs[0].transpose(1, 2, 0)
            imgs[int(y) : int(y)+ int(p_h), int(x) : int(x)+ int(p_w)] = r, g, b
            imgs[int(y): int(y) + int(p_h), int(x): int(x) + int(p_w)] = \
                np.flip(imgs[int(y) : int(y)+ int(p_h), int(x) : int(x)+ int(p_w)])
            img = torch.tensor(imgs).unsqueeze(0).permute(0, 3, 1, 2).to(self.device)

            #im = im.to(self.device)
        return img