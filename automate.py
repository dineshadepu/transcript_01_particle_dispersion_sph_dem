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


# n_core = 6
n_core = 16
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


class Wu2014FallingSolid3D(Problem):
    """We will allow a circular particle to settle in a tank with different
    viscosity coefficients and check the deviatioinn in the settling time
    """
    def get_name(self):
        return 'wu_2014_falling_solid_3d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/wu_2014_falling_solid_3d.py' + backend

        # kinetic viscosity
        # Base case info
        self.case_info = {
            'N_8': (dict(
                N=8,
                nu=1e-6,
                ), 'N=8'),

            'N_10': (dict(
                N=10,
                nu=1e-6,
                ), 'N=10'),

            'N_12': (dict(
                N=12,
                nu=1e-6,
                ), 'N=12'),
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
        self.plot_displacement()

    def plot_displacement(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        rand_case = (list(data.keys())[0])

        t_exp = data[rand_case]['t_exp']
        z_position_exp = data[rand_case]['z_position_exp']

        # ==================================
        # Plot x amplitude
        # ==================================
        plt.plot(t_exp, z_position_exp, "o", label='Experiment')
        for name in self.case_info:
            t_current = data[name]['t_current']
            z_current = data[name]['z_position_current']

            plt.plot(t_current, z_current, label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('z - amplitude')
        plt.legend()
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('z_vs_t.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot x amplitude
        # ==================================

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


class Zhang2009SolidFluidMixtureVerification2d(Problem):
    """

    SPH-VCPM-DEM paper has all the details of the DEM model
    https://www.sciencedirect.com/science/article/pii/S0889974621001523#sec3
    """
    def get_name(self):
        return 'zhang_2009_solid_fluid_mixture_verification_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/zhang_2009_solid_fluid_mixture_verification_2d.py' + backend

        # kinetic viscosity
        # Base case info
        self.case_info = {
            'N_8': (dict(
                N=8,
                fric_coeff=0.1,
                fric_coeff_wall=0.1,
                en=0.999,
                en_wall=0.85,
                ), 'N=8'),

            'N_10': (dict(
                N=10,
                fric_coeff=0.1,
                fric_coeff_wall=0.1,
                en=0.999,
                en_wall=0.85,
                ), 'N=10'),

            'N_12': (dict(
                N=12,
                fric_coeff=0.1,
                fric_coeff_wall=0.1,
                en=0.999,
                en_wall=0.85,
                ), 'N=12'),
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


class Dinesh2024MixingWithStirrerHomogeneous2d(Problem):
    """

    Own example
    """
    def get_name(self):
        return 'dinesh_2024_mixing_with_stirrer_homogeneous_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dinesh_2024_mixing_with_stirrer_2d.py' + backend

        # kinetic viscosity
        # Base case info
        self.case_info = {
            'case_1': (dict(
                stirrer_velocity=1.
                ), 'Case 1'),

            'case_2': (dict(
                stirrer_velocity=3.
                ), 'Case 2'),

            'case_3': (dict(
                stirrer_velocity=5.
                ), 'Case 3'),

            'main_case': (dict(
                stirrer_velocity=0.3
                ), 'Main case'),
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


class Dinesh2024MixingWithStirrerInHomogeneous2d(Problem):
    """

    Own example
    """
    def get_name(self):
        return 'dinesh_2024_mixing_with_stirrer_inhomogeneous_2d'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/dinesh_2024_mixing_with_stirrer_2d.py' + backend

        # kinetic viscosity
        # Base case info
        self.case_info = {
            'case_1': (dict(
                stirrer_velocity=1.,
                radius_ratio=1.2,
                ), 'Case 1'),

            'case_2': (dict(
                stirrer_velocity=3.,
                radius_ratio=1.2,
                ), 'Case 2'),

            'main_case': (dict(
                stirrer_velocity=0.3
                ), 'Main case'),

            # 'case_3': (dict(
            #     stirrer_velocity=5.,
            #     radius_ratio=1.5,
            #     ), 'Case 3'),
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


class RFCTesting1(Problem):
    """We will allow a circular particle to settle in a tank with different
    viscosity coefficients and check the deviatioinn in the settling time
    """
    def get_name(self):
        return 'rfc_testing_1'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/wu_2014_falling_solid_2d.py' + backend

        # kinetic viscosity
        # Base case info
        self.case_info = {
            'nu_0': (dict(
                nu=0.,
                ), 'nu=0.'),

            'nu_6': (dict(
                nu=1e-6,
                ), 'nu=1e-6'),

            'nu_5': (dict(
                nu=1e-5,
                ), 'nu=1e-5'),

            'nu_4': (dict(
                nu=1e-4,
                ), 'nu=1e-4'),

            'nu_3': (dict(
                nu=1e-3,
                ), 'nu=1e-3')
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
        self.plot_displacement()

    def plot_displacement(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        rand_case = (list(data.keys())[0])

        t_exp = data[rand_case]['t_exp']
        y_position_exp = data[rand_case]['y_position_exp']

        # ==================================
        # Plot x amplitude
        # ==================================
        plt.plot(t_exp, y_position_exp, "o", label='Experiment')
        for name in self.case_info:
            t_current = data[name]['t_current']
            y_current = data[name]['y_position_current']

            plt.plot(t_current, y_current, label=self.case_info[name][1])

        plt.xlabel('time')
        plt.ylabel('y - amplitude')
        plt.legend()
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('y_vs_t.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot x amplitude
        # ==================================

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


class Hashemi2012FallingCircularCylinder(Problem):
    """We will allow a circular particle to settle in a tank with different
    viscosity coefficients and check the deviatioinn in the settling time
    """
    def get_name(self):
        return 'hashemi_2012_falling_of_circular_cylinder_in_a_closed_channel'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/hashemi_2012_falling_of_circular_cylinder_in_a_closed_channel.py' + backend

        # kinetic viscosity
        # Base case info
        self.case_info = {
            'N_10': (dict(
                N=10,
                nu=1e-2,
                ), 'N=10'),

            'N_8': (dict(
                N=8,
                nu=1e-2,
                ), 'N=8'),
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
        self.plot_displacement()

    def plot_displacement(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        rand_case = (list(data.keys())[0])

        t_zhang_FPM_PST = data[rand_case]['t_zhang_FPM_PST']
        vertical_position_zhang_FPM_PST = data[rand_case]['vertical_position_zhang_FPM_PST']
        t_hashemi_sph = data[rand_case]['t_hashemi_sph']
        velocity_hashemi_SPH = data[rand_case]['velocity_hashemi_SPH']

        # ==================================
        # Plot x amplitude
        # ==================================
        plt.plot(t_zhang_FPM_PST, vertical_position_zhang_FPM_PST, "o", label='Zhang et al. 2019 (FPM-PST)')
        for name in self.case_info:
            t_current = data[name]['t_current']
            z_current = data[name]['y_position_current']

            plt.plot(t_current, z_current, label="Current " + str(self.case_info[name][1]))

        plt.xlabel('time')
        plt.ylabel('y - position')
        plt.legend()
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('y_vs_t.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot x amplitude
        # ==================================

        # ==================================
        # Plot x amplitude
        # ==================================
        plt.plot(t_hashemi_sph, velocity_hashemi_SPH, "o", label='Hashemi et al. 2012 (SPH)')
        for name in self.case_info:
            t_current = data[name]['t_current']
            z_current = data[name]['v_velocity_current']

            plt.plot(t_current, z_current, label="Current " + str(self.case_info[name][1]))

        plt.xlabel('time')
        plt.ylabel('Vertical velocity')
        plt.legend()
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('v_vs_t.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot x amplitude
        # ==================================

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


class Hashemi2012DKT(Problem):
    """
    """
    def get_name(self):
        return 'hashemi_2012_DKT'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/hashemi_2012_dkt.py' + backend

        # kinetic viscosity
        # Base case info
        self.case_info = {
            'N_10': (dict(
                N=10,
                nu=1e-2,
                ), 'N=10'),

            'N_8': (dict(
                N=8,
                nu=1e-2,
                ), 'N=8'),
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
        self.plot_displacement()

    def plot_displacement(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        rand_case = (list(data.keys())[0])

        t_zhang_FPM_PST = data[rand_case]['t_zhang_FPM_PST']
        vertical_position_zhang_FPM_PST = data[rand_case]['vertical_position_zhang_FPM_PST']
        t_hashemi_sph = data[rand_case]['t_hashemi_sph']
        velocity_hashemi_SPH = data[rand_case]['velocity_hashemi_SPH']

        # ==================================
        # Plot x amplitude
        # ==================================
        plt.plot(t_zhang_FPM_PST, vertical_position_zhang_FPM_PST, "o", label='Zhang et al. 2019 (FPM-PST)')
        for name in self.case_info:
            t_current = data[name]['t_current']
            z_current = data[name]['y_position_current']

            plt.plot(t_current, z_current, label="Current " + str(self.case_info[name][1]))

        plt.xlabel('time')
        plt.ylabel('y - position')
        plt.legend()
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('y_vs_t.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot x amplitude
        # ==================================

        # ==================================
        # Plot x amplitude
        # ==================================
        plt.plot(t_hashemi_sph, velocity_hashemi_SPH, "o", label='Hashemi et al. 2012 (SPH)')
        for name in self.case_info:
            t_current = data[name]['t_current']
            z_current = data[name]['v_velocity_current']

            plt.plot(t_current, z_current, label="Current " + str(self.case_info[name][1]))

        plt.xlabel('time')
        plt.ylabel('Vertical velocity')
        plt.legend()
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('v_vs_t.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot x amplitude
        # ==================================

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


class Hashemi2012DKTVariableRadius(Problem):
    """
    """
    def get_name(self):
        return 'hashemi_2012_DKT_variable_radius'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/hashemi_2012_dkt.py' + backend

        # kinetic viscosity
        # Base case info
        self.case_info = {
            'top': (dict(
                N=10,
                nu=1e-2,
                radius_ratio=1.2,
                top=None,
                ), 'Top Bigger'),

            'bottom': (dict(
                N=8,
                nu=1e-2,
                radius_ratio=1.2,
                no_top=None,
                ), 'Bottom Bigger'),
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


class Hashemi2012DKTParallel(Problem):
    """
    """
    def get_name(self):
        return 'hashemi_2012_DKT_parallel'

    def setup(self):
        get_path = self.input_path

        cmd = 'python code/hashemi_2012_dkt.py' + backend

        # kinetic viscosity
        # Base case info
        self.case_info = {
            'left_big': (dict(
                N=8,
                nu=1e-2,
                radius_ratio=1.2,
                parallel_arrangement=None,
                no_top=None
                ), 'Left big'),

            'right_big': (dict(
                N=8,
                nu=1e-2,
                radius_ratio=1.2,
                parallel_arrangement=None,
                top=None
                ), 'Right big'),
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
        self.plot_displacement()

    def plot_displacement(self):
        data = {}
        for name in self.case_info:
            data[name] = np.load(self.input_path(name, 'results.npz'))

        rand_case = (list(data.keys())[0])

        t_zhang_FPM_PST = data[rand_case]['t_zhang_FPM_PST']
        vertical_position_zhang_FPM_PST = data[rand_case]['vertical_position_zhang_FPM_PST']
        t_hashemi_sph = data[rand_case]['t_hashemi_sph']
        velocity_hashemi_SPH = data[rand_case]['velocity_hashemi_SPH']

        # ==================================
        # Plot x amplitude
        # ==================================
        plt.plot(t_zhang_FPM_PST, vertical_position_zhang_FPM_PST, "o", label='Zhang et al. 2019 (FPM-PST)')
        for name in self.case_info:
            t_current = data[name]['t_current']
            z_current = data[name]['y_position_current']

            plt.plot(t_current, z_current, label="Current " + str(self.case_info[name][1]))

        plt.xlabel('time')
        plt.ylabel('y - position')
        plt.legend()
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('y_vs_t.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot x amplitude
        # ==================================

        # ==================================
        # Plot x amplitude
        # ==================================
        plt.plot(t_hashemi_sph, velocity_hashemi_SPH, "o", label='Hashemi et al. 2012 (SPH)')
        for name in self.case_info:
            t_current = data[name]['t_current']
            z_current = data[name]['v_velocity_current']

            plt.plot(t_current, z_current, label="Current " + str(self.case_info[name][1]))

        plt.xlabel('time')
        plt.ylabel('Vertical velocity')
        plt.legend()
        # plt.tight_layout(pad=0)
        plt.savefig(self.output_path('v_vs_t.pdf'))
        plt.clf()
        plt.close()
        # ==================================
        # Plot x amplitude
        # ==================================

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
        ParticleDispersion3D,
        # Problem  no 4
        # Wu 2014 falling solid in water 3d,
        Wu2014FallingSolid3D,

        # Problem  no 5
        # Zhang 2009 solid-fluid mixture verification
        Zhang2009SolidFluidMixtureVerification2d,

        # Problem  no 5
        # Own problem
        Dinesh2024MixingWithStirrerHomogeneous2d,
        Dinesh2024MixingWithStirrerInHomogeneous2d,

        # Problem  no 6
        Hashemi2012FallingCircularCylinder,

        # Problem  no 8
        Hashemi2012DKT,

        # Problem  no 9
        Hashemi2012DKTVariableRadius,

        # Problem  no 9
        Hashemi2012DKTParallel,

        # RFC testing no 1
        RFCTesting1
        ]

    automator = Automator(
        simulation_dir='outputs',
        output_dir=os.path.join('manuscript', 'figures'),
        all_problems=PROBLEMS
    )
    automator.run()
