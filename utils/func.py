# -*- coding: utf-8 -*-
# Original code Copyright (c) 2023 CyberAgent AI Lab
# Modifications Copyright (c) 2025 Hyungtak Lee
# Licensed under the MIT License.
from abs import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union, Iterable
from typing_extensions import TypedDict

import numpy as np
import numpy.typing as npt

## TODO: Better shape/boundary checks for input other than atleast_2d
##    - input needs to be the exact dimensions.
##    - fidelity should be in the range.

class TestFunc:
    __metaclass__ = ABCMeta

    def mf_values(self, input_list: List[npt.ArrayLike]) -> List[npt.ArrayLike]:
        """
        return each fidelity output list

        Parameters:
        -----------
            input_list : list of numpy array
            list size is the number of fidelity M
            each numpy array size is (N_m \times d), N_m is the number of data of each fidelity.

        Returns:
        --------
            output_list : list of numpy array
            each numpy array size is (N_m, 1)
        """
        func_values_list: List = []
        for m in range(len(input_list)):
            if np.size(input_list[m]) != 0:
                func_values_list.append(self.values(input_list[m], fidelity=m))
            else:
                func_values_list.append(np.array([]))

        return func_values_list

    def mf_costs(self, input_list: List[npt.ArrayLike]) -> List[npt.ArrayLike]:
        """
        return each fidelity cost list depending on x

        Parameters:
        -----------
            input_list : list of numpy array
            list size is the number of fidelity M
            each numpy array size is (N_m \times d), N_m is the number of data of each fidelity.

        Returns:
        --------
            output_list : list of numpy array
            each numpy array size is (N_m, 1)
        """
        func_costs_list = []
        for m in range(len(input_list)):
            if np.size(input_list[m]) != 0:
                func_costs_list.append(self.costs(input_list[m], fidelity=m))
            else:
                func_costs_list.append(np.array([]))
        return func_costs_list

    @abstractmethod
    def values(self, input: npt.ArrayLike, fidelity: Optional[int] = None) -> npt.ArrayLike:
        pass

    @abstractmethod
    def costs(self, input: npt.ArrayLike, fidelity: Optional[int] = None) -> npt.ArrayLike:
        pass


def standard_length_scale(bounds: npt.ArrayLike) -> npt.ArrayLike:
    return (bounds[1] - bounds[0]) / 2.0


class Beale(TestFunc):
    """
    Beale Function: d = 2, M = 2
    Three constants is changed to make low fidelity function.
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[-4.5, -4.5], [4.5, 4.5]])
        self.d = 2
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = 0

    def values(self, input: npt.ArrayLike, fidelity: Optional[int] = None) -> npt.ArrayLike:
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._common_processing(input, cons_list=[1.2, 2.5, 2.5])
        elif fidelity == 1:
            return self._common_processing(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _common_processing(self, input: npt.ArrayLike, cons_list: Optional[List[float]] = None) -> npy.ArrayLike:
        if cons_list is None:
            cons_list = [1.5, 2.25, 2.625]
        first_term = (cons_list[0] - input[:, 0] + input[:, 0] * input[:, 1]) ** 2
        second_term = (cons_list[1] - input[:, 0] + input[:, 0] * input[:, 1] ** 2) ** 2
        third_term = (cons_list[2] - input[:, 0] + input[:, 0] * input[:, 1] ** 3) ** 2
        return -np.c_[(first_term + second_term + third_term)]


class HartMann3(TestFunc):
    """
    HartMann 3-dimensional function: d = 3, M = 3
    The alpha is minused constant (fidelity=0->0.2, fidelity=1->0.1) to make low fidelity functions.
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0, 0], [1, 1, 1]])
        self.d = 3
        self.M = 3
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = 3.86278

    def values(self, input: npt.ArrayLike, fidelity: Optional[int] = None) -> npt.ArrayLike:
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._common_processing(input, alpha_error=0.2)
        elif fidelity == 1:
            return self._common_processing(input, alpha_error=0.1)
        elif fidelity == 2:
            return self._common_processing(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _common_processing(self, input: npt.ArrayLike, alpha_error: float = 0) -> npt.ArrayLike:
        alpha = np.array([1.0, 1.2, 3.0, 3.2]) - alpha_error
        A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
        P = (
            np.array(
                [
                    [3689, 1170, 2673],
                    [4699, 4387, 7470],
                    [1091, 8732, 5547],
                    [381, 5743, 8828],
                ]
            )
            * 1e-4
        )

        values = 0
        for i in range(4):
            inner = 0
            for j in range(3):
                inner -= A[i, j] * (input[:, j] - P[i, j]) ** 2
            values += alpha[i] * np.power(np.e, inner)
        return np.c_[values]


class HartMann4(TestFunc):
    """
    HartMann 4-dimensional function: d = 4, M = 3
    The alpha is minused constant (fidelity=0->0.2, fidelity=1->0.1) to make low fidelity functions.
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
        self.d = 4
        self.M = 3
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = 3.135474

    def values(self, input: npt.ArrayLike, fidelity: Optional[int] = None) -> npt.ArrayLike:
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._common_processing(input, alpha_error=0.2)
        elif fidelity == 1:
            return self._common_processing(input, alpha_error=0.1)
        elif fidelity == 2:
            return self._common_processing(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _common_processing(self, input: npt.ArrayLike, alpha_error: float = 0) -> npt.ArrayLike:
        alpha = np.array([1.0, 1.2, 3.0, 3.2]) - alpha_error
        A = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        P = (
            np.array(
                [
                    [1312, 1696, 5569, 124, 8283, 5886],
                    [2329, 4135, 8307, 3736, 1004, 9991],
                    [2348, 1451, 3522, 2883, 3047, 6650],
                    [4047, 8828, 8732, 5743, 1091, 381],
                ]
            )
            * 1e-4
        )

        values = 0
        for i in range(4):
            inner = 0
            for j in range(4):
                inner -= A[i, j] * (input[:, j] - P[i, j]) ** 2
            values += alpha[i] * np.power(np.e, inner)
        return np.c_[(values - 1.1) / 0.839]


class Borehole(TestFunc):
    """
    Borehole function: d = 8, M = 2
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array(
            [
                [0.05, 100, 63070, 990, 63.1, 700, 1120, 9855],
                [0.15, 50000, 115600, 1110, 116, 820, 1680, 12045],
            ]
        )
        self.d = 8
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = (
            -7.8198
        )  # On Efficient Global Optimization via Universal Kriging Surrogate Models

    def values(self, input: npt.ArrayLike, fidelity: Optional[int] = None) -> npt.ArrayLike:
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._low_fidelity_values(input)
        elif fidelity == 1:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input: npt.ArrayLike) -> npt.ArrayLike:
        numerator = 2 * np.pi * input[:, 2] * (input[:, 3] - input[:, 5])
        log_ratio = np.log(input[:, 1] / input[:, 0])
        denominator = log_ratio * (
            1
            + (2 * input[:, 6] * input[:, 2])
            / (log_ratio * input[:, 0] ** 2 * input[:, 7])
            + input[:, 2] / input[:, 4]
        )
        values = numerator / denominator
        return -np.c_[values]

    def _low_fidelity_values(self, input: npt.ArrayLike) -> npt.ArrayLike:
        numerator = 5 * input[:, 2] * (input[:, 3] - input[:, 5])
        log_ratio = np.log(input[:, 1] / input[:, 0])
        denominator = log_ratio * (
            1.5
            + (2 * input[:, 6] * input[:, 2])
            / (log_ratio * input[:, 0] ** 2 * input[:, 7])
            + input[:, 2] / input[:, 4]
        )
        values = numerator / denominator
        return -np.c_[values]


class Branin(TestFunc):
    """
    Branin function: d = 2, M = 2
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[-5, 0], [10, 15]])
        self.d = 2
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = -0.397887

    def values(self, input: npt.ArrayLike, fidelity: Optional[int] = None) -> npt.ArrayLike:
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._low_fidelity_values(input)
        elif fidelity == 1:
            return self._high_fidelity_values(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _high_fidelity_values(self, input: npt.ArrayLike) -> npt.ArrayLike:
        a = 1
        b = 5.1 / (4 * np.pi**2)
        c = 5 / np.pi
        r = 6
        s = 10
        t = 1 / (8 * np.pi)
        first_term = a * (input[:, 1] - b * input[:, 0] ** 2 + c * input[:, 0] - r) ** 2
        second_term = s * (1 - t) * np.cos(input[:, 0])
        return -np.c_[(first_term + second_term + s)]

    def _low_fidelity_values(self, input: npt.ArrayLike) -> npt.ArrayLike:
        a = 1.1
        b = 5.0 / (4 * np.pi**2)
        c = 4 / np.pi
        r = 5
        s = 8
        t = 1 / (10 * np.pi)
        first_term = a * (input[:, 1] - b * input[:, 0] ** 2 + c * input[:, 0] - r) ** 2
        second_term = s * (1 - t) * np.cos(input[:, 0])
        return -np.c_[(first_term + second_term + s)]


class Colville(TestFunc):
    """
    Colville function: d = 4, M = 2
    low_fidelity function is
    """

    def __init__(self):
        self.noise_var = 0
        self.bounds = np.array([[-10, -10, -10, -10], [10, 10, 10, 10]])
        self.d = 4
        self.M = 2
        self.standard_length_scale = standard_length_scale(self.bounds)
        self.maximum = 0

    def values(self, input: npt.ArrayLike, fidelity: Optional[int] = None):
        """
        return output of entered fidelity

        Parameters:
        -----------
            input : numpy array (N_m \times d)
            fidelity : int

        Returns:
        --------
            output : numpy array (N_m, 1)
        """
        input = np.atleast_2d(input)
        if fidelity is None:
            fidelity = self.M - 1
        if fidelity == 0:
            return self._common_processing(input, cons_list=[90, 0.9, 0.9, 100, 9, 20])
        elif fidelity == 1:
            return self._common_processing(input)
        else:
            print("Not implemented fidelity")
            exit(1)

    def _common_processing(self, input: npt.ArrayLike, cons_list: Optional[List[float]] = None) -> npt.ArrayLike:
        if cons_list is None:
            cons_list = [100, 1, 1, 90, 10.1, 19.8]
        term1 = cons_list[0] * (input[:, 0] ** 2 - input[:, 1]) ** 2
        term2 = cons_list[1] * (input[:, 0] - 1) ** 2
        term3 = cons_list[2] * (input[:, 2] - 1) ** 2
        term4 = cons_list[3] * (input[:, 2] ** 2 - input[:, 3]) ** 2
        term5 = cons_list[4] * ((input[:, 1] - 1) ** 2 + (input[:, 3] - 1) ** 2)
        term6 = cons_list[5] * (input[:, 1] - 1) * (input[:, 3] - 1)
        return -np.c_[(term1 + term2 + term3 + term4 + term5 + term6)]
