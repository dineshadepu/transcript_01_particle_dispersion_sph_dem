#!/usr/bin/env python
import os
import matplotlib.pyplot as plt

from itertools import cycle, product
import json
from automan.api import PySPHProblem as Problem
from automan.api import Automator, Simulation, filter_by_name
from automan.jobs import free_cores
from pysph.solver.utils import load, get_files
from automan.api import (Automator, Simulation, filter_cases, filter_by_name)

import numpy as np
import matplotlib
matplotlib.use('agg')
from cycler import cycler
from matplotlib import rc, patches, colors
from matplotlib.collections import PatchCollection

rc('font', **{'family': 'Helvetica', 'size': 12})
rc('legend', fontsize='medium')
rc('axes', grid=True, linewidth=1.2)
rc('axes.grid', which='both', axis='both')
# rc('axes.formatter', limits=(1, 2), use_mathtext=True, min_exponent=1)
rc('grid', linewidth=0.5, linestyle='--')
rc('xtick', direction='in', top=True)
rc('ytick', direction='in', right=True)
rc('savefig', format='pdf', bbox='tight', pad_inches=0.05,
   transparent=False, dpi=300)
rc('lines', linewidth=1.5)
rc('axes', prop_cycle=(
    cycler('color', ['tab:blue', 'tab:green', 'tab:red',
                     'tab:orange', 'm', 'tab:purple',
                     'tab:pink', 'tab:gray']) +
    cycler('linestyle', ['-.', '--', '-', ':',
                         (0, (3, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1)),
                         (0, (3, 2, 1, 1)), (0, (3, 2, 2, 1, 1, 1)),
                         ])
))


# n_core = 32
n_core = 6
n_thread = n_core * 2
backend = ' --openmp '


def pretty(d, indent=0):
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            pretty(value, indent + 1)
        else:
            print('\t' * (indent + 1) + str(value))


def scheme_opts(params):
    if isinstance(params, tuple):
        return params[0]
    return params


def get_files_at_given_times(files, times):
    from pysph.solver.utils import load
    result = []
    count = 0
    for f in files:
        data = load(f)
        t = data['solver_data']['t']
        if count >= len(times):
            break
        if abs(t - times[count]) < t * 1e-8:
            result.append(f)
            count += 1
        elif t > times[count]:
            result.append(f)
            count += 1
    return result


def get_files_at_given_times_from_log(files, times, logfile):
    import re
    result = []
    time_pattern = r"output at time\ (\d+(?:\.\d+)?)"
    file_count, time_count = 0, 0
    with open(logfile, 'r') as f:
        for line in f:
            if time_count >= len(times):
                break
            t = re.findall(time_pattern, line)
            if t:
                if float(t[0]) in times:
                    result.append(files[file_count])
                    time_count += 1
                elif float(t[0]) > times[time_count]:
                    result.append(files[file_count])
                    time_count += 1
                file_count += 1
    return result


class ManuscriptRFCImageGenerator(Problem):
    """
    Pertains to Figure 14 (a)
    """
    def get_name(self):
        return 'manuscript_rfc_image_generator'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/manuscript_rfc_image_generator.py' + backend


        # Base case info
        self.case_info = {
            'case_1': (dict(
            ), 'Case 1'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.move_figures()

    def move_figures(self):
        import shutil
        import os

        source = "code/manuscript_rfc_image_generator_output/"

        target_dir = "manuscript/figures/manuscript_rfc_image_generator_output/"
        os.makedirs(target_dir, exist_ok=True)
        # print(target_dir)

        file_names = os.listdir(source)

        for file_name in file_names:
            # print(file_name)
            if file_name.endswith((".jpg", ".pdf", ".png")):
                # print(target_dir)
                shutil.copy(os.path.join(source, file_name), target_dir)


class DamBreak(Problem):
    """
    Pertains to Figure 14 (a)
    """
    def get_name(self):
        return 'dam_break'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dam_break.py' + backend

        velocity = 3.9
        fric_coeff = 0.092
        dt = 1e-7
        # Base case info
        self.case_info = {
            'db_2d_N_10': (dict(
                dim=2,
                N=10,
                timestep=dt,
                ), 'N=10'),

            'db_2d_N_20': (dict(
                dim=2,
                N=20,
                timestep=dt,
                ), 'N=20'),

            'db_3d_N_10': (dict(
                dim=3,
                N=10,
                timestep=dt,
                ), 'N=10'),

            'db_3d_N_20': (dict(
                dim=3,
                N=20,
                timestep=dt,
                ), 'N=20'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       velocity=velocity,
                       fric_coeff=fric_coeff,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.move_figures()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


class SphericalBodiesSettlingInTankDEM2D(Problem):
    """
    Pertains to Figure 14 (a)
    """
    def get_name(self):
        return 'spherical_bodies_settling_in_tank_DEM_2D'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/spherical_bodies_settling_in_tank_DEM_2D.py' + backend

        # Base case info
        self.case_info = {
            'point_collision': (dict(
                N=10,
                ), 'N=10'),

            'surface_collision': (dict(
                N=10,
                ), 'N=10'),

        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.move_figures()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


class SphericalBodiesSettlingInTankDEM3D(Problem):
    """
    Pertains to Figure 14 (a)
    """
    def get_name(self):
        return 'spherical_bodies_settling_in_tank_DEM_3D'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/spherical_bodies_settling_in_tank_DEM_3D.py' + backend

        # Base case info
        self.case_info = {
            'point_collision': (dict(
                N=10,
                ), 'N=10'),

            'surface_collision': (dict(
                N=10,
                ), 'N=10'),
        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.move_figures()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


class ParticleDispersion2D(Problem):
    """
    Pertains to Figure 14 (a)
    """
    def get_name(self):
        return 'particle_dispersion_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/particle_dispersion_2d.py' + backend

        # Base case info
        self.case_info = {
            'point_collision': (dict(
                N=10,
                ), 'N=10'),

            'surface_collision': (dict(
                N=10,
                ), 'N=10'),

        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.move_figures()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


class ParticleDispersion3D(Problem):
    """
    Pertains to Figure 14 (a)
    """
    def get_name(self):
        return 'particle_dispersion_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/particle_dispersion_3d.py' + backend

        # Base case info
        self.case_info = {
            'point_collision': (dict(
                N=10,
                ), 'N=10'),

            'surface_collision': (dict(
                N=10,
                ), 'N=10'),

        }

        self.cases = [
            Simulation(get_path(name), cmd,
                       job_info=dict(n_core=n_core,
                                     n_thread=n_thread), cache_nnps=None,
                       **scheme_opts(self.case_info[name][0]))
            for name in self.case_info
        ]

    def run(self):
        self.make_output_dir()
        self.move_figures()

    def move_figures(self):
        import shutil
        import os

        for name in self.case_info:
            source = self.input_path(name)

            target_dir = "manuscript/figures/" + source[8:] + "/"
            os.makedirs(target_dir)
            # print(target_dir)

            file_names = os.listdir(source)

            for file_name in file_names:
                # print(file_name)
                if file_name.endswith((".jpg", ".pdf", ".png")):
                    # print(target_dir)
                    shutil.copy(os.path.join(source, file_name), target_dir)


if __name__ == '__main__':
    PROBLEMS = [
        # Image generator
        ManuscriptRFCImageGenerator,

        # Problem  no 1 (Dam break 2d and 3d)
        DamBreak,
        # Problem  no 2
        SphericalBodiesSettlingInTankDEM2D,
        SphericalBodiesSettlingInTankDEM3D,
        # Problem  no 3
        ParticleDispersion2D,
        ParticleDispersion3D
        # # Problem  no 4
        # Hashemi2012NeutrallyBuoyantCircularCylinderInShearFlow,
        # # Problem  no 5
        # Ng2021TwoCylindersInShearFlow,
        ]

    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()
