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
from sph_dem.geometry import (create_tank_2d_from_block_2d)

from sph_dem.rigid_fluid_coupling import (
    ParticlesFluidScheme, add_rigid_fluid_properties_to_rigid_body)

from sph_dem.rigid_body.rigid_body_3d import (setup_rigid_body,
                                              set_linear_velocity,
                                              set_angular_velocity,
                                              get_master_and_slave_rb,
                                              add_contact_properties_body_master)

from pysph_dem.dem_simple import (setup_wall_dem)


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


class Problem(Application):
    def add_user_options(self, group):
        group.add_argument("--fluid-length-ratio", action="store", type=float,
                           dest="fluid_length_ratio", default=40,
                           help="Ratio between the fluid length and rigid body diameter")

        group.add_argument("--fluid-height-ratio", action="store", type=float,
                           dest="fluid_height_ratio", default=8,
                           help="Ratio between the fluid height and rigid body diameter")

        group.add_argument("--tank-length-ratio", action="store", type=float,
                           dest="tank_length_ratio", default=1,
                           help="Ratio between the tank length and fluid length")

        group.add_argument("--tank-height-ratio", action="store", type=float,
                           dest="tank_height_ratio", default=4.,
                           help="Ratio between the tank height and fluid height")

        group.add_argument("--N", action="store", type=int, dest="N",
                           default=6,
                           help="Number of particles in diamter of a rigid cylinder")

        group.add_argument("--rigid-body-rho", action="store", type=float,
                           dest="rigid_body_rho", default=1500,
                           help="Density of rigid cylinder")

        group.add_argument("--rigid-body-diameter", action="store", type=float,
                           dest="rigid_body_diameter", default=1e-3,
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
        self.rigid_body_diameter = 1. * 1e-2
        self.rigid_body_velocity = 0.
        self.no_of_layers = self.options.no_of_layers
        self.no_of_bodies = 3 * 6 * self.options.no_of_layers

        # x - axis
        self.fluid_length = 7.1 * 1e-2
        # y - axis
        self.fluid_height = 12.1 * 1e-2
        # z - axis
        self.fluid_depth = 0.

        # x - axis
        self.tank_length = 26 * 1e-2
        # y - axis
        self.tank_height = 14. * 1e-2
        # z - axis
        self.tank_depth = 0.0

        self.tank_layers = 3

        # x - axis
        self.stirrer_length = self.fluid_length * 0.1
        # y - axis
        self.stirrer_height = self.fluid_height * 1.2
        # z - axis
        self.stirrer_depth = self.fluid_depth * 0.5
        self.stirrer_velocity = 0.
        # time period of stirrer
        # self.T = (self.stirrer_length * 3) / self.stirrer_velocity

        # ======================
        # Dimensions ends
        # ======================

        # ======================
        # Physical properties and consants
        # ======================
        self.fluid_rho = 1000.
        self.rigid_body_rho = 2700
        self.rigid_body_E = 69e9
        self.rigid_body_nu = 0.3
        self.wall_body_rho = 2700
        self.wall_body_E = 3. * 1e9
        self.wall_body_nu = 0.3
        self.fric_coeff_body_body = 0.1
        self.fric_coeff_body_wall = 0.1
        self.cor_body_body = 0.9999
        self.cor_body_wall = 0.85

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
        self.wall_time = 0.3
        self.tf = self.wall_time + 0.6
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
        self.output_at_times = [0., self.wall_time+0.1, self.wall_time+0.3, self.wall_time+0.5]
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

        xt, yt = create_tank_2d_from_block_2d(xf, yf, self.tank_length, self.tank_height,
                                              self.dx, 3)
        zt = np.zeros_like(xt)
        zf = np.zeros_like(xf)

        # move fluid such that the left corner is at the origin of the
        # co-ordinate system
        translation = translate_system_with_left_corner_as_origin(xf, yf, zf)
        xt[:] = xt[:] - translation[0]
        yt[:] = yt[:] - translation[1]
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

    def create_six_spherical_particles_in_a_row(self):
        x1, y1 = create_circle_1(self.rigid_body_diameter, self.dx)
        x2, y2 = create_circle_1(self.rigid_body_diameter, self.dx)
        x2[:] = x1[:] + self.rigid_body_diameter + self.dx
        x3, y3 = create_circle_1(self.rigid_body_diameter, self.dx)
        x3[:] = x2[:] + self.rigid_body_diameter + self.dx
        x4, y4 = create_circle_1(self.rigid_body_diameter, self.dx)
        x4[:] = x3[:] + self.rigid_body_diameter + self.dx
        x5, y5 = create_circle_1(self.rigid_body_diameter, self.dx)
        x5[:] = x4[:] + self.rigid_body_diameter + self.dx
        x6, y6 = create_circle_1(self.rigid_body_diameter, self.dx)
        x6[:] = x5[:] + self.rigid_body_diameter + self.dx

        x = np.concatenate((x1, x2, x3, x4, x5, x6))
        y = np.concatenate((y1, y2, y3, y4, y5, y6))
        return x, y

    def create_five_spherical_particles_in_a_row(self):
        x1, y1 = create_circle_1(self.rigid_body_diameter, self.dx)
        x2, y2 = create_circle_1(self.rigid_body_diameter, self.dx)
        x2[:] = x1[:] + self.rigid_body_diameter + self.dx
        x3, y3 = create_circle_1(self.rigid_body_diameter, self.dx)
        x3[:] = x2[:] + self.rigid_body_diameter + self.dx
        x4, y4 = create_circle_1(self.rigid_body_diameter, self.dx)
        x4[:] = x3[:] + self.rigid_body_diameter + self.dx
        x5, y5 = create_circle_1(self.rigid_body_diameter, self.dx)
        x5[:] = x4[:] + self.rigid_body_diameter + self.dx
        x6, y6 = create_circle_1(self.rigid_body_diameter, self.dx)
        x6[:] = x5[:] + self.rigid_body_diameter + self.dx

        x = np.concatenate((x1, x2, x3, x4, x5))
        y = np.concatenate((y1, y2, y3, y4, y5))

        x[:] += self.rigid_body_diameter / 2.
        return x, y

    def create_rb_geometry_particle_array_without_gaps(self):
        x_six_1, y_six_1 = self.create_six_spherical_particles_in_a_row()
        x_five_1, y_five_1 = self.create_five_spherical_particles_in_a_row()
        y_five_1[:] += self.rigid_body_diameter - 1. * self.dx

        x_six_2, y_six_2 = self.create_six_spherical_particles_in_a_row()
        x_five_2, y_five_2 = self.create_five_spherical_particles_in_a_row()
        y_five_2[:] += self.rigid_body_diameter - 1. * self.dx

        y_six_2[:] += 2 * self.rigid_body_diameter - 2. * self.dx
        y_five_2[:] += 2 * self.rigid_body_diameter - 2. * self.dx

        x_six_3, y_six_3 = self.create_six_spherical_particles_in_a_row()
        x_five_3, y_five_3 = self.create_five_spherical_particles_in_a_row()
        y_five_3[:] += self.rigid_body_diameter + self.dx

        y_six_3[:] += 4 * self.rigid_body_diameter - 4. * self.dx
        y_five_3[:] += 4 * self.rigid_body_diameter - 6. * self.dx

        x = np.concatenate((x_six_1, x_six_2, x_six_3, x_five_1, x_five_2, x_five_3))
        y = np.concatenate((y_six_1, y_six_2, y_six_3, y_five_1, y_five_2, y_five_3))
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

        body_id = np.array([])
        dem_id = np.array([])
        total_no_of_bodies = 6 * 3 + 5 * 3
        for i in range(total_no_of_bodies):
            body_id = np.concatenate((body_id, i * np.ones_like(x_circle,
                                                                dtype='int')))
            dem_id = np.concatenate((dem_id, i * np.ones_like(x_circle,
                                                              dtype='int')))

        rigid_body_combined.add_property('body_id', type='int', data=body_id)
        rigid_body_combined.add_property('dem_id', type='int', data=dem_id)
        return rigid_body_combined

    def create_rb_geometry_particle_array(self):
        x_six_1, y_six_1 = self.create_six_spherical_particles_in_a_row()
        x_five_1, y_five_1 = self.create_five_spherical_particles_in_a_row()
        y_five_1[:] += self.rigid_body_diameter + 1. * self.dx

        x_six_2, y_six_2 = self.create_six_spherical_particles_in_a_row()
        x_five_2, y_five_2 = self.create_five_spherical_particles_in_a_row()
        y_five_2[:] += self.rigid_body_diameter + 1. * self.dx

        y_six_2[:] += 2 * self.rigid_body_diameter + 2. * self.dx
        y_five_2[:] += 2 * self.rigid_body_diameter + 2. * self.dx

        x_six_3, y_six_3 = self.create_six_spherical_particles_in_a_row()
        x_five_3, y_five_3 = self.create_five_spherical_particles_in_a_row()
        y_five_3[:] += self.rigid_body_diameter + self.dx

        y_six_3[:] += 4 * self.rigid_body_diameter + 4. * self.dx
        y_five_3[:] += 4 * self.rigid_body_diameter + 4. * self.dx

        x = np.concatenate((x_six_1, x_six_2, x_six_3, x_five_1, x_five_2, x_five_3))
        y = np.concatenate((y_six_1, y_six_2, y_six_3, y_five_1, y_five_2, y_five_3))
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

        body_id = np.array([])
        dem_id = np.array([])
        total_no_of_bodies = 6 * 3 + 5 * 3
        for i in range(total_no_of_bodies):
            body_id = np.concatenate((body_id, i * np.ones_like(x_circle,
                                                                dtype='int')))
            dem_id = np.concatenate((dem_id, i * np.ones_like(x_circle,
                                                              dtype='int')))

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
                                              u=self.stirrer_velocity,
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
        rigid_body_combined = self.create_rb_geometry_particle_array_without_gaps()
        # rigid_body_combined.x[:] -= min(fluid.x) - min(rigid_body_combined.x)
        # move it to right, so that we can have a separate view
        disp_x = 0.
        # rigid_body_combined.y[:] += self.rigid_body_diameter * 1.
        rigid_body_combined.y[:] += min(fluid.y) - min(rigid_body_combined.y[:])
        rigid_body_combined.x[:] += min(fluid.x) - min(rigid_body_combined.x[:])
        rigid_body_combined.x[:] += self.dx

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
            lin_vel = np.concatenate((lin_vel, np.array([sign * 0., 0., 0.])))
            ang_vel = np.concatenate((ang_vel, np.array([0., 0., 0.])))

        # set_linear_velocity(rigid_body_combined, lin_vel)
        # set_angular_velocity(rigid_body_combined, ang_vel)

        # This is # 3,
        rigid_body_master, rigid_body_slave = get_master_and_slave_rb(
            rigid_body_combined
        )

        # This is # 4
        rigid_body_master.rad_s[:] = self.rigid_body_diameter / 2.
        rigid_body_master.h[:] = self.rigid_body_diameter * 2.
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
        x[:] += disp_x
        y = np.array([max(tank.y) / 2.,
                      max(tank.y) / 2.,
                      min(tank.y) + (self.tank_layers - 1) * self.dx
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
                                             E=3. * 1e9,
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
        stirrer.x[:] += max(fluid.x) - min(stirrer.x)
        # stirrer.x[:] -= self.rigid_body_diameter
        stirrer.y[:] -= min(stirrer.y) - min(fluid.y)
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

        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=200,
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
        T = self.wall_time
        if t < T:
            for pa in self.particles:
                if pa.name == 'stirrer':
                    pa.y[:] += 2. * dt

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
            'zhang_2009_solid_fluid_mixture_verification_2d.log')
        to_plot = get_files_at_given_times_from_log(output_files, output_times,
                                                    logfile)
        for i, f in enumerate(to_plot):
            data = load(f)
            _t = data['solver_data']['t']
            fluid = data['arrays']['fluid']
            tank = data['arrays']['tank']
            rigid_body = data['arrays']['rigid_body_combined_slave']

            c_min = min((fluid.u[:]**2. + fluid.v[:]**2. + fluid.w[:]**2.)**0.5)
            c_max = max((fluid.u[:]**2. + fluid.v[:]**2. + fluid.w[:]**2.)**0.5)
            vmag = (fluid.u[:]**2. + fluid.v[:]**2. + fluid.w[:]**2.)**0.5
            # c_min = min(fluid.vmag)
            # c_max = max(fluid.vmag)

            s = 5
            fig, axs = plt.subplots(1, 1, figsize=(12, 6))

            axs.scatter(tank.x, tank.y, s=1, c="k")
            axs.scatter(rigid_body.x, rigid_body.y, s=1, c="r")
            tmp = axs.scatter(fluid.x, fluid.y, s=1, c=vmag, vmin=c_min,
                              vmax=c_max, cmap="jet")
            # tmp = axs.scatter(fluid.x, fluid.y, s=s, c=fluid.p, vmin=c_min,
            #                   vmax=c_max, cmap="hot")

            axs.set_xlabel('x')
            axs.set_ylabel('y')
            # axs.set_xlim([x_min, x_max])
            # axs.set_ylim([y_min, y_max])
            # axs.grid()
            axs.set_aspect('equal', 'box')

            divider = make_axes_locatable(axs)
            cax = divider.append_axes('right', size='3%', pad=0.1)
            fig.colorbar(tmp, cax=cax, format='%.0e', orientation='vertical',
                         shrink=0.3)
            cax.set_ylabel('Velocity magnitude')  # cax == cb.ax

            # save the figure
            figname = os.path.join(os.path.dirname(fname), "time" + str(i) + ".pdf")
            fig.savefig(figname, dpi=300)


if __name__ == '__main__':
    app = Problem()
    app.run()
    app.post_process(app.info_filename)
