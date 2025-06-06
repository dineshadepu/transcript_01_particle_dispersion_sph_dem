"""Rigid body falling in tank but 2D

Run it using:

python rb_falling_in_hs_tank.py --openmp --pfreq 100 --timestep 1e-4 --alpha 0.1 --fric-coeff 0.05 --en 0.05 --N 5 --no-of-bodies 1 --fluid-length-ratio 5 --fluid-height-ratio 10 --rigid-body-rho 2000 --tf 3 --scheme combined -d rb_falling_in_hs_tank_combined_output

python rb_falling_in_hs_tank.py --openmp --pfreq 100 --timestep 1e-4 --alpha 0.1 --fric-coeff 0.05 --en 0.05 --N 5 --no-of-bodies 1 --fluid-length-ratio 5 --fluid-height-ratio 10 --rigid-body-rho 2000 --tf 3 --scheme master -d rb_falling_in_hs_tank_master_output


TODO:

1. Create many particles
2. Run it on HPC
3. Change the speed
4. Change the dimensions
5. Commit SPH-DEM and current repository
6. Validate 1 spherical particle settled in hs tank
7. Validate 2 spherical particles settled in hs tank
8. Complete the manuscript
9. Different particle diameter
10. Different density
"""
import numpy as np
import sys

from pysph.examples import dam_break_2d as DB
from pysph.tools.geometry import get_2d_tank, get_2d_block
from pysph.tools.geometry import get_3d_sphere
import pysph.tools.geometry as G
from pysph.base.utils import get_particle_array

from pysph.examples import cavity as LDC
from pysph.sph.equation import Equation, Group
from pysph.sph.scheme import add_bool_argument
sys.path.insert(0, "./../")
from pysph_rfc_new.fluids import (get_particle_array_fluid, get_particle_array_boundary)

from pysph.base.kernels import (QuinticSpline)
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.scheme import SchemeChooser

from pysph_dem.rigid_body.boundary_particles import (add_boundary_identification_properties,
                                get_boundary_identification_etvf_equations)
from pysph_rfc_new.geometry import hydrostatic_tank_2d, create_circle_1, translate_system_with_left_corner_as_origin
# from geometry import hydrostatic_tank_2d, create_circle_1

from sph_dem.rigid_fluid_coupling import (
    ParticlesFluidScheme, add_rigid_fluid_properties_to_rigid_body)

from sph_dem.rigid_body.rigid_body_3d import (setup_rigid_body,
                                              set_linear_velocity,
                                              set_angular_velocity,
                                              get_master_and_slave_rb,
                                              add_contact_properties_body_master)

from pysph_dem.dem_simple import (setup_wall_dem)
from zhang_2009_solid_fluid_mixture_verification_2d import (get_files_at_given_times_from_log)


def check_time_make_zero(t, dt):
    if t < 0.0:
        return True
    else:
        return False


class MakeForcesZeroOnRigidBody(Equation):
    def initialize(self, d_idx, d_fx, d_fy, d_fz):
        d_fx[d_idx] = 0.
        d_fy[d_idx] = 0.
        d_fz[d_idx] = 0.

    def reduce(self, dst, t, dt):
        frc = declare('object')
        trq = declare('object')

        frc = dst.force
        trq = dst.torque

        frc[:] = 0
        trq[:] = 0


class Problem(Application):
    def add_user_options(self, group):
        group.add_argument("--fluid-length-ratio", action="store", type=float,
                           dest="fluid_length_ratio", default=30,
                           help="Ratio between the fluid length and rigid body diameter")

        group.add_argument("--fluid-height-ratio", action="store", type=float,
                           dest="fluid_height_ratio", default=15,
                           help="Ratio between the fluid height and rigid body diameter")

        group.add_argument("--tank-length-ratio", action="store", type=float,
                           dest="tank_length_ratio", default=1,
                           help="Ratio between the tank length and fluid length")

        group.add_argument("--tank-height-ratio", action="store", type=float,
                           dest="tank_height_ratio", default=3.,
                           help="Ratio between the tank height and fluid height")

        group.add_argument("--N", action="store", type=int, dest="N",
                           default=6,
                           help="Number of particles in diamter of a rigid cylinder")

        group.add_argument("--rigid-body-rho", action="store", type=float,
                           dest="rigid_body_rho", default=1050,
                           help="Density of rigid cylinder")

        group.add_argument("--rigid-body-diameter", action="store", type=float,
                           dest="rigid_body_diameter", default=1e-3,
                           help="Diameter of each particle")

        group.add_argument("--stirrer-velocity", action="store", type=float,
                           dest="stirrer_velocity", default=0.319,
                           help="Stirrer velocity")

        group.add_argument("--radius-ratio", action="store", type=float,
                           dest="radius_ratio", default=1.0,
                           help="Diameter of each particle")

        add_bool_argument(
            group,
            'follow-combined-rb-solver',
            dest='follow_combined_rb_solver',
            default=True,
            help='Use combined particle array solver for rigid body dynamics')

        group.add_argument("--no-of-layers", action="store", type=int,
                           dest="no_of_layers", default=2,
                           help="Total no of rigid bodies layers")

    def consume_user_options(self):
        # ======================
        # Get the user options and save them
        # ======================
        # self.re = self.options.re
        # ======================
        # ======================

        # ======================
        # Dimensions
        # ======================
        # dimensions rigid body dimensions
        # All the particles are in circular or spherical shape
        self.rigid_body_diameter = 0.11
        self.radius_ratio = self.options.radius_ratio
        self.rigid_body_velocity = 0.
        self.no_of_layers = self.options.no_of_layers
        self.no_of_bodies = 18 * 3

        # x - axis
        self.fluid_length = self.options.fluid_length_ratio * self.rigid_body_diameter
        # y - axis
        self.fluid_height = self.options.fluid_height_ratio * self.rigid_body_diameter
        # z - axis
        self.fluid_depth = 0.

        # x - axis
        self.tank_length = self.options.tank_length_ratio * self.fluid_length
        # y - axis
        self.tank_height = self.options.tank_height_ratio * self.fluid_height
        # z - axis
        self.tank_depth = 0.0

        self.tank_layers = 3

        # x - axis
        self.stirrer_length = self.rigid_body_diameter
        # y - axis
        self.stirrer_height = self.fluid_height * 0.5
        # z - axis
        self.stirrer_depth = self.fluid_depth * 0.5

        # stirrer velocity
        self.stirrer_velocity = self.options.stirrer_velocity
        # time period of stirrer
        self.T = (self.fluid_length - 6. * self.stirrer_length) / self.stirrer_velocity
        # self.T = (self.stirrer_length) / self.stirrer_velocity

        # ======================
        # Dimensions ends
        # ======================

        # ======================
        # Physical properties and consants
        # ======================
        self.fluid_rho = 1000.
        self.rigid_body_rho = self.options.rigid_body_rho
        self.rigid_body_E = 1e9
        self.rigid_body_nu = 0.23

        self.gx = 0.
        self.gy = -9.81
        self.gz = 0.
        self.dim = 2
        # ======================
        # Physical properties and consants ends
        # ======================

        # ======================
        # Numerical properties
        # ======================
        self.hdx = 1.
        self.N = self.options.N
        self.dx = self.rigid_body_diameter / self.N
        self.h = self.hdx * self.dx
        self.vref = np.sqrt(2. * abs(self.gy) * self.fluid_height)
        self.c0 = 10 * self.vref
        self.mach_no = self.vref / self.c0
        self.nu = 0.0
        self.wall_time = 1.0
        self.tf = self.wall_time + 10
        # self.tf = 0.56 - 0.3192
        self.p0 = self.fluid_rho*self.c0**2
        self.alpha = 0.00

        # Setup default parameters.
        dt_cfl = 0.25 * self.h / (self.c0 + self.vref)
        dt_viscous = 1e5
        if self.nu > 1e-12:
            dt_viscous = 0.125 * self.h**2/self.nu
        dt_force = 0.25 * np.sqrt(self.h/(np.abs(self.gy)))
        print("dt_cfl", dt_cfl, "dt_viscous", dt_viscous, "dt_force", dt_force)

        self.dt = min(dt_cfl, dt_force)
        print("Computed stable dt is: ", self.dt)
        self.total_steps = self.tf / self.dt
        print("Total steps in this simulation are", self.total_steps)
        self.pfreq = int(self.total_steps / 300)
        print("Pfreq is", self.pfreq)
        self.output_at_times = [0., self.wall_time+3.0, self.wall_time+6.0, self.wall_time+9.9]
        # self.dt = 1e-4
        # ==========================
        # Numerical properties ends
        # ==========================

        # ==========================
        # Numerical properties ends
        # ==========================
        self.follow_combined_rb_solver = self.options.follow_combined_rb_solver
        # ==========================
        # Numerical properties ends
        # ==========================

    def create_fluid_and_tank_particle_arrays(self):
        xf, yf, xt, yt = hydrostatic_tank_2d(self.fluid_length, self.fluid_height,
                                             self.tank_height, self.tank_layers,
                                             self.dx, self.dx, False)
        zf = np.zeros_like(xf)
        # move fluid such that the left corner is at the origin of the
        # co-ordinate system
        translation = translate_system_with_left_corner_as_origin(xf, yf, zf)
        xt[:] = xt[:] - translation[0]
        yt[:] = yt[:] - translation[1]

        xt_1, yt_1 = get_2d_block(dx=self.dx,
                                  length=self.fluid_length,
                                  height=self.stirrer_length)
        xt_1 += min(xf) - min(xt_1)
        yt_1 += max(yt) - max(yt_1)
        xt = np.concatenate([xt, xt_1])
        yt = np.concatenate([yt, yt_1])

        zt = np.zeros_like(xt)

        zt[:] = zt[:] - translation[2]

        # xf, yf, zf = np.array([0.02]), np.array([self.fluid_height]), np.array([0.])

        m = self.dx**self.dim * self.fluid_rho
        fluid = get_particle_array_fluid(name='fluid', x=xf, y=yf, z=zf, h=self.h, m=m, rho=self.fluid_rho)
        tank = get_particle_array_boundary(name='tank', x=xt, y=yt, z=zt, h=self.h, m=m, rho=self.fluid_rho)

        # set the pressure of the fluid
        fluid.p[:] = - self.fluid_rho * self.gy * (max(fluid.y) - fluid.y[:])
        fluid.c0_ref[0] = self.c0
        fluid.p0_ref[0] = self.p0
        return fluid, tank

    def create_nine_bodies(self, diameter):
        x1, y1 = create_circle_1(diameter, self.dx)
        x2, y2 = create_circle_1(diameter, self.dx)
        x3, y3 = create_circle_1(diameter, self.dx)

        x2 += diameter + self.dx * 2
        x3 += 2. * diameter + self.dx * 4

        # x2 += x1 + self.rigid_body_diameter + self.dx * 2
        # x3 += x2 + self.rigid_body_diameter + self.dx * 2

        x_three_bottom = np.concatenate((x1, x2, x3))
        y_three_bottom = np.concatenate((y1, y2, y3))

        x_three_middle = np.copy(x_three_bottom)
        y_three_middle = np.copy(y_three_bottom)
        y_three_middle += diameter + self.dx * 2

        x_three_top = np.copy(x_three_middle)
        y_three_top = np.copy(y_three_middle)
        y_three_top += diameter + self.dx * 2

        x = np.concatenate((x_three_bottom, x_three_middle, x_three_top))
        y = np.concatenate((y_three_bottom, y_three_middle, y_three_top))

        body_id = np.array([])
        dem_id = np.array([])
        for i in range(9):
            body_id = np.concatenate((body_id, i * np.ones_like(x1,
                                                                dtype='int')))
            dem_id = np.concatenate((dem_id, i * np.ones_like(x1,
                                                              dtype='int')))

        return x, y, body_id, dem_id

    def create_rb_geometry_particle_array(self):
        x_1, y_1, body_id_1, dem_id_1 = self.create_nine_bodies(self.rigid_body_diameter)
        rad_1 = np.ones(9) * self.rigid_body_diameter / 2.

        x_2, y_2, body_id_2, dem_id_2 = self.create_nine_bodies(self.rigid_body_diameter * self.radius_ratio)
        x_2 += self.dx * -1.
        y_2 += max(y_1) - min(y_2) + self.rigid_body_diameter * 1.
        rad_2 = np.ones(9) * (self.rigid_body_diameter * self.radius_ratio) / 2.
        body_id_2 += 9
        dem_id_2 += 9

        x = np.concatenate((x_1, x_2))
        y = np.concatenate((y_1, y_2))
        rad = np.concatenate((rad_1, rad_2))
        body_id = np.concatenate((body_id_1, body_id_2))
        dem_id = np.concatenate((dem_id_1, dem_id_2))

        x_right, y_right = np.copy(x), np.copy(y)
        x_right += 7 * self.rigid_body_diameter
        rad_right = np.concatenate((rad_1, rad_2))
        body_id_right = np.copy((body_id)) + 18
        dem_id_right = np.copy((dem_id)) + 18

        x_right_2, y_right_2 = np.copy(x_right), np.copy(y_right)
        x_right_2 += 7 * self.rigid_body_diameter
        rad_right_2 = np.concatenate((rad_1, rad_2))
        body_id_right_2 = np.copy((body_id_right)) + 18
        dem_id_right_2 = np.copy((dem_id_right)) + 18

        x = np.concatenate((x, x_right, x_right_2))
        y = np.concatenate((y, y_right, y_right_2))
        self.rad_master = np.concatenate((rad, rad_right, rad_right_2))
        body_id = np.concatenate((body_id, body_id_right, body_id_right_2))
        dem_id = np.concatenate((dem_id, dem_id_right, dem_id_right_2))
        # import matplotlib.pyplot as plt
        # plt.scatter(x, y)
        # plt.show()

        y[:] += self.fluid_height + self.rigid_body_diameter
        # x[:] += self.fluid_length/2. + self.rigid_body_diameter
        x[:] += self.fluid_length/2. - self.rigid_body_diameter * 4.
        x[:] += self.rigid_body_diameter * 4.
        z = np.zeros_like(x)

        m = self.rigid_body_rho * self.dx**self.dim * np.ones_like(x)
        h = self.h
        rad_s = self.dx / 2.
        # This is # 1
        rigid_body_combined = get_particle_array(name='rigid_body_combined',
                                                 x=x,
                                                 y=y,
                                                 z=z,
                                                 h=1.2 * h,
                                                 m=m,
                                                 E=1e9,
                                                 nu=0.23,
                                                 rho=self.fluid_rho)

        x_circle, y_circle = create_circle_1(self.rigid_body_diameter, self.dx)

        rigid_body_combined.add_property('body_id', type='int', data=body_id)
        rigid_body_combined.add_property('dem_id', type='int', data=dem_id)
        return rigid_body_combined

    def create_stirrer(self):
        x_stirrer, y_stirrer = get_2d_block(dx=self.dx,
                                            length=self.stirrer_length,
                                            height=self.stirrer_height)
        m = self.dx**self.dim * self.fluid_rho
        stirrer = get_particle_array_boundary(name='stirrer',
                                              x=x_stirrer, y=y_stirrer,
                                              u=0.,
                                              h=self.h, m=m,
                                              rho=self.fluid_rho,
                                              E=1e9,
                                              nu=0.3,
                                              G=1e9,
                                              rad_s=self.dx)
        dem_id = np.ones_like(x_stirrer, dtype='int')
        stirrer.add_property('dem_id', type='int', data=dem_id)
        return stirrer

    def create_particles(self):
        # This will create full particle array required for the scheme
        fluid, tank = self.create_fluid_and_tank_particle_arrays()

        # =========================
        # create rigid body
        # =========================
        # Steps in creating the the right rigid body
        # 1. Create a particle array
        # 2. Get the combined rigid body with properties computed
        # 3. Seperate out center of mass and particles into two particle arrays
        # 4. Add dem contact properties to the master particle array
        # 5. Add rigid fluid coupling properties to the slave particle array
        # Some important notes.
        # 1. Make sure the mass is consistent for all the equations, since
        # we use 'm_b' for some equations and 'm' for fluid coupling
        rigid_body_combined = self.create_rb_geometry_particle_array()
        # move rigid body to left
        rigid_body_combined.x[:] -= min(rigid_body_combined.x) - min(fluid.x)
        rigid_body_extent = max(rigid_body_combined.x) - min(rigid_body_combined.x)
        fluid_extent = max(fluid.x) - min(fluid.x)
        rigid_body_center_length = rigid_body_extent * 0.5

        rigid_body_combined.x[:] += (fluid_extent - rigid_body_extent) * 0.5

        # This is # 2, (Here we create a rigid body which is compatible with
        # combined rigid body solver formulation)
        setup_rigid_body(rigid_body_combined, self.dim)
        rigid_body_combined.h[:] = self.h
        rigid_body_combined.rad_s[:] = self.dx / 2.

        # print("moi body ", body.inertia_tensor_inverse_global_frame)
        lin_vel = np.array([])
        ang_vel = np.array([])
        sign = -1
        for i in range(self.no_of_bodies):
            sign *= -1
            if i == 6:
                lin_vel = np.concatenate((lin_vel, np.array([sign * 0., 0., 0.])))
                ang_vel = np.concatenate((ang_vel, np.array([0., 0., 0.])))
            else:
                lin_vel = np.concatenate((lin_vel, np.array([sign * 0., 0., 0.])))
                ang_vel = np.concatenate((ang_vel, np.array([0., 0., 0.])))

        set_linear_velocity(rigid_body_combined, lin_vel)
        set_angular_velocity(rigid_body_combined, ang_vel)

        # This is # 3,
        rigid_body_master, rigid_body_slave = get_master_and_slave_rb(
            rigid_body_combined
        )

        # This is # 4
        rigid_body_master.rad_s[:] = self.rad_master[:]
        rigid_body_master.h[:] = self.rigid_body_diameter * 0.5
        add_contact_properties_body_master(rigid_body_master, 6, 3)

        # This is # 5
        add_rigid_fluid_properties_to_rigid_body(rigid_body_slave)
        # set mass and density to correspond to fluid
        rigid_body_slave.m[:] = self.fluid_rho * self.dx**2.
        rigid_body_slave.rho[:] = self.fluid_rho
        # similarly for combined rb particle arra
        add_rigid_fluid_properties_to_rigid_body(rigid_body_combined)
        # set mass and density to correspond to fluid
        rigid_body_combined.m[:] = self.fluid_rho * self.dx**2.
        rigid_body_combined.rho[:] = self.fluid_rho

        # =========================
        # create rigid body ends
        # =========================

        # ======================
        # create wall for rigid body
        # ======================
        # left right bottom
        x = np.array([min(tank.x) + self.tank_layers * self.dx,
                      max(tank.x) - self.tank_layers * self.dx,
                      max(tank.x) / 2
                     ])
        # x[:] += disp_x
        y = np.array([max(tank.y) / 2.,
                      max(tank.y) / 2.,
                      min(tank.y) + self.tank_layers * self.dx
                      ])
        normal_x = np.array([1., -1., 0.])
        normal_y = np.array([0., 0., 1.])
        normal_z = np.array([0., 0., 0.])
        rigid_body_wall = get_particle_array(name='rigid_body_wall',
                                             x=x,
                                             y=y,
                                             normal_x=normal_x,
                                             normal_y=normal_y,
                                             normal_z=normal_z,
                                             h=self.rigid_body_diameter/2.,
                                             rho_b=self.rigid_body_rho,
                                             rad_s=self.rigid_body_diameter/2.,
                                             E=69. * 1e9,
                                             nu=0.3,
                                             G=69. * 1e5)
        dem_id = np.array([0, 0, 0])
        rigid_body_wall.add_property('dem_id', type='int', data=dem_id)
        rigid_body_wall.add_constant('no_wall', [3])
        setup_wall_dem(rigid_body_wall)

        # remove fluid particles overlapping with the rigid body
        G.remove_overlap_particles(
            fluid, rigid_body_combined, self.dx, dim=self.dim
        )
        # remove fluid particles overlapping with the rigid body
        G.remove_overlap_particles(
            fluid, rigid_body_slave, self.dx, dim=self.dim
        )

        # Add stirrer
        stirrer = self.create_stirrer()
        # set the position of the stirrer
        stirrer.x[:] += min(fluid.x) - min(stirrer.x) + self.stirrer_length * 3
        # stirrer.x[:] -= self.rigid_body_diameter
        stirrer.y[:] += ((min(fluid.y) - min(stirrer.y)) +
                         self.fluid_height) - self.stirrer_height * 0.5
        # stirrer.y[:] -= self.rigid_body_diameter
        G.remove_overlap_particles(
            fluid, stirrer, self.dx, dim=self.dim
        )

        return [fluid, tank, rigid_body_master, rigid_body_slave,
                rigid_body_wall, stirrer]

    def create_scheme(self):
        master = ParticlesFluidScheme(
            fluids=['fluid'],
            boundaries=['tank', "stirrer"],
            # rigid_bodies_combined=[],
            rigid_bodies_master=["rigid_body_combined_master"],
            rigid_bodies_slave=["rigid_body_combined_slave"],
            rigid_bodies_wall=["rigid_body_wall"],
            stirrer=["stirrer"],
            dim=2,
            rho0=0.,
            h=0.,
            c0=0.,
            pb=0.,
            nu=0.,
            gy=0.,
            alpha=0.)

        s = SchemeChooser(default='master', master=master)
        return s

    def configure_scheme(self):
        tf = self.tf
        scheme = self.scheme
        scheme.configure(
            dim=self.dim,
            rho0=self.fluid_rho,
            h=self.h,
            c0=self.c0,
            pb=self.p0,
            nu=self.nu,
            gy=self.gy)

        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=self.pfreq,
                                output_at_times=self.output_at_times)
        print("dt = %g"%self.dt)

    # def create_equations(self):
    #     # print("inside equations")
    #     eqns = self.scheme.get_equations()

    #     # Apply external force
    #     zero_frc = []
    #     zero_frc.append(
    #         MakeForcesZeroOnRigidBody("rigid_body", sources=None))

    #     # print(eqns.groups)
    #     eqns.groups[-1].append(Group(equations=zero_frc,
    #                                  condition=check_time_make_zero))

    #     return eqns

    def post_step(self, solver):
        t = solver.t
        dt = solver.dt
        for pa in self.particles:
            if pa.name == 'stirrer':
                if t > self.wall_time:
                    if ((t - self.wall_time) // self.T) % 2 == 0:
                        pa.u[:] = self.stirrer_velocity
                        pa.x[:] += pa.u[:] * dt
                    else:
                        pa.u[:] = -self.stirrer_velocity
                        pa.x[:] += pa.u[:] * dt
                else:
                    pa.u[:] = 0.

    def customize_output(self):
        self._mayavi_config('''
        # b = particle_arrays['rigid_body']
        # b.scalar = 'm'
        b = particle_arrays['fluid']
        b.scalar = 'vmag'
        ''')

    def post_process(self, fname):

        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from pysph.solver.utils import load, get_files
        from pysph.solver.utils import iter_output
        import os

        info = self.read_info(fname)
        files = self.output_files

        # print(files)
        info = self.read_info(fname)
        output_files = self.output_files
        output_times = self.output_at_times
        logfile = os.path.join(
            os.path.dirname(fname),
            'dinesh_2024_mixing_with_stirrer_2d.log')
        to_plot = get_files_at_given_times_from_log(output_files, output_times,
                                                    logfile)
        for i, f in enumerate(to_plot):
            data = load(f)
            _t = data['solver_data']['t']
            fluid = data['arrays']['fluid']
            tank = data['arrays']['tank']
            stirrer = data['arrays']['stirrer']
            rigid_body = data['arrays']['rigid_body_combined_slave']
            rigid_body_master = data['arrays']['rigid_body_combined_master']

            c_min = min((fluid.u[:]**2. + fluid.v[:]**2. + fluid.w[:]**2.)**0.5)
            c_max = max((fluid.u[:]**2. + fluid.v[:]**2. + fluid.w[:]**2.)**0.5)
            vmag = (fluid.u[:]**2. + fluid.v[:]**2. + fluid.w[:]**2.)**0.5
            # c_min = min(fluid.vmag)
            # c_max = max(fluid.vmag)

            s = 0.1
            fig, axs = plt.subplots(1, 1, figsize=(12, 6))

            axs.scatter(tank.x, tank.y, s=s, c="k")
            axs.scatter(stirrer.x, stirrer.y, s=s, c="k")
            axs.scatter(rigid_body.x, rigid_body.y, s=s, c="r")
            # # =====================================================
            # # Color the rigid body particles based on the radius
            # # =====================================================
            # if len(rigid_body.x) < 1804:
            #     axs.scatter(rigid_body.x, rigid_body.y, s=s, c="r")
            # else:
            #     self.dx = 0.11 / 6
            #     x1, y1 = create_circle_1(0.11, self.dx)
            #     small_radius_indices = np.array([])
            #     for k in range(27):
            #         small_radius_indices = np.concatenate([small_radius_indices, np.zeros_like(x1)])

            #     x1, y1 = create_circle_1(1.2 * 0.11, self.dx)
            #     big_radius_indices = np.array([])
            #     for k in range(27):
            #         big_radius_indices = np.concatenate([big_radius_indices, np.ones_like(x1)])
            #     all_indices = np.concatenate([small_radius_indices, big_radius_indices])
            #     small_indices = np.where(all_indices==0.)[0]
            #     axs.scatter(rigid_body.x[small_indices], rigid_body.y[small_indices], s=s, color="red")

            #     big_indices = np.where(all_indices==1.)[0]
            #     axs.scatter(rigid_body.x[big_indices], rigid_body.y[big_indices], s=s, color="black")


            # # print(all_indices)

            # # get the body ids which have same radius


            # # max_body_id = max(rigid_body.body_id)
            # # min_body_id = min(rigid_body.body_id)
            # # print("maximum bid is ", max_body_id)
            # # print("minim bid is ", min_body_id)
            # # max_body_id_indices = np.where(rigid_body.body_id==max_body_id)[0]
            # # min_body_id_indices = np.where(rigid_body.body_id==min_body_id)[0]
            # # print(max_body_id_indices)
            # # print(min_body_id_indices)
            # # print(len(max_body_id_indices))
            # # print(len(min_body_id_indices))
            # # if (len(max_body_id_indices) == len(min_body_id_indices)):
            # #     axs.scatter(rigid_body.x, rigid_body.y, s=s, c="r")
            # # =====================================================
            # # Color the rigid body particles based on the radius
            # # =====================================================
            if i!=0:
                tmp = axs.scatter(fluid.x, fluid.y, s=s, c=vmag, vmin=c_min,
                                  vmax=c_max, cmap="jet", rasterized=True)
            else:
                axs.scatter(fluid.x, fluid.y, s=s, c=vmag, vmin=c_min,
                            vmax=c_max, cmap="jet", rasterized=True)
            # tmp = axs.scatter(fluid.x, fluid.y, s=s, c=fluid.p, vmin=c_min,
            #                   vmax=c_max, cmap="hot")

            axs.set_xlabel('x (m)')
            axs.set_ylabel('y (m)')
            # axs.set_xlim([x_min, x_max])
            # axs.set_ylim([y_min, y_max])
            # axs.grid()
            axs.set_aspect('equal', 'box')

            if i != 0:
                divider = make_axes_locatable(axs)
                cax = divider.append_axes('right', size='3%', pad=0.1)
                fig.colorbar(tmp, cax=cax, format='%.0e', orientation='vertical',
                            shrink=0.3)
                cax.set_ylabel('Velocity magnitude (m/s)')  # cax == cb.ax

            # save the figure
            print(str(i), "string printable")
            figname = os.path.join(os.path.dirname(fname), "time" + str(i) + ".pdf")
            fig.savefig(figname, dpi=300, bbox_inches='tight', pad_inches=0.05)


if __name__ == '__main__':
    app = Problem()
    app.run()
    app.post_process(app.info_filename)
