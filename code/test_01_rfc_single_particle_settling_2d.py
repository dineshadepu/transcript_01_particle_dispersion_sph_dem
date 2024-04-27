"""Rigid body falling in tank but 2D

Run it using:

python rb_falling_in_hs_tank.py --openmp --pfreq 100 --timestep 1e-4 --alpha 0.1 --fric-coeff 0.05 --en 0.05 --N 5 --no-of-bodies 1 --fluid-length-ratio 5 --fluid-height-ratio 10 --rigid-body-rho 2000 --tf 3 --scheme combined -d rb_falling_in_hs_tank_combined_output

python rb_falling_in_hs_tank.py --openmp --pfreq 100 --timestep 1e-4 --alpha 0.1 --fric-coeff 0.05 --en 0.05 --N 5 --no-of-bodies 1 --fluid-length-ratio 5 --fluid-height-ratio 10 --rigid-body-rho 2000 --tf 3 --scheme master -d rb_falling_in_hs_tank_master_output


TODO:

1. Close the tank
2. Remove the stirrer
3. Get the dimentions of the tank right
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
                           default=8,
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
        self.rigid_body_diameter = 0.125
        self.rigid_body_velocity = 0.
        self.no_of_layers = self.options.no_of_layers
        self.no_of_bodies = 3 * 6 * self.options.no_of_layers

        # x - axis
        self.fluid_length = 5 * self.rigid_body_diameter
        # y - axis
        self.fluid_height = 20. * self.rigid_body_diameter
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
        self.stirrer_length = self.fluid_length * 0.1
        # y - axis
        self.stirrer_height = self.fluid_height * 0.5
        # z - axis
        self.stirrer_depth = self.fluid_depth * 0.5
        self.stirrer_velocity = 1.
        # time period of stirrer
        self.T = (self.stirrer_length * 3) / self.stirrer_velocity

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
        print("With N", self.N, "the dx value is", self.dx)
        print("With delta as in hashemi 2012 paper 1 / 6000.", 1 / 6000., "the dx value is same.")
        print("With delta as in hashemi 2012 paper 1 / 10000.", 1 / 10000., "the dx value is same.")
        print("With delta as in hashemi 2012 paper 1 / 15000.", 1 / 15000., "the dx value is same.")
        self.h = self.hdx * self.dx
        self.vref = np.sqrt(2. * abs(self.gy) * self.fluid_height)
        self.c0 = 10 * self.vref
        self.mach_no = self.vref / self.c0
        self.nu = 0.0
        self.tf = 0.8
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
        # self.dt = 1e-5
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
        # xf, yf, xt, yt = hydrostatic_tank_2d(self.fluid_length, self.fluid_height,
        #                                      self.tank_height, self.tank_layers,
        #                                      self.dx, self.dx, True)
        xf, yf, xt, yt = hydrostatic_tank_2d(self.fluid_length, self.fluid_height,
                                             self.fluid_height*1.1, self.tank_layers,
                                             self.dx, self.dx, False)

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

    def create_six_bodies(self):
        x1, y1 = create_circle_1(self.rigid_body_diameter, self.dx)
        x2, y2 = create_circle_1(self.rigid_body_diameter, self.dx)
        x3, y3 = create_circle_1(self.rigid_body_diameter, self.dx)

        x2 += self.rigid_body_diameter + self.dx * 2
        x3 += 2. * self.rigid_body_diameter + self.dx * 4

        # x2 += x1 + self.rigid_body_diameter + self.dx * 2
        # x3 += x2 + self.rigid_body_diameter + self.dx * 2

        x_three_bottom = np.concatenate((x1, x2, x3))
        y_three_bottom = np.concatenate((y1, y2, y3))

        x_three_top = np.copy(x_three_bottom)
        y_three_top = np.copy(y_three_bottom)
        y_three_top += self.rigid_body_diameter + self.dx * 2

        x = np.concatenate((x_three_bottom, x_three_top))
        y = np.concatenate((y_three_bottom, y_three_top))
        return x, y

    def create_rb_geometry_particle_array(self):
        x, y = create_circle_1(self.rigid_body_diameter, self.dx)
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
        for i in range(1):
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
        rigid_body_combined = self.create_rb_geometry_particle_array()
        rigid_body_extent = max(rigid_body_combined.x) - min(rigid_body_combined.x)
        rigid_body_combined.x[:] -= min(rigid_body_combined.x) - min(fluid.x)
        rigid_body_combined.x[:] += self.rigid_body_diameter
        rigid_body_combined.x[:] += rigid_body_extent / 2.
        # move it to right, so that we can have a separate view
        disp_x = 0.
        rigid_body_combined.y[:] += max(fluid.y) - max(rigid_body_combined.y)
        rigid_body_combined.y[:] -= 2. * self.rigid_body_diameter
        rigid_body_combined.x[:] -= min(rigid_body_combined.x) - min(fluid.x)
        rigid_body_combined.x[:] += self.fluid_length / 2. - self.rigid_body_diameter / 2.

        # This is # 2, (Here we create a rigid body which is compatible with
        # combined rigid body solver formulation)
        setup_rigid_body(rigid_body_combined, self.dim)
        rigid_body_combined.h[:] = self.h
        rigid_body_combined.rad_s[:] = self.dx / 2.

        # print("moi body ", body.inertia_tensor_inverse_global_frame)
        lin_vel = np.array([])
        ang_vel = np.array([])
        sign = -1
        for i in range(1):
            sign *= -1
            lin_vel = np.concatenate((lin_vel, np.array([sign * 0., 0., 0.])))
            ang_vel = np.concatenate((ang_vel, np.array([0., 0., 0.])))

        set_linear_velocity(rigid_body_combined, lin_vel)
        set_angular_velocity(rigid_body_combined, ang_vel)

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
        stirrer.x[:] += ((min(fluid.x) - min(stirrer.x)) +
                         self.fluid_length * 0.5) - self.stirrer_length * 0.5
        # stirrer.x[:] -= self.rigid_body_diameter
        stirrer.y[:] += max(fluid.y) - min(stirrer.y) + self.rigid_body_diameter * 10
        G.remove_overlap_particles(
            fluid, stirrer, self.dx, dim=self.dim
        )

        return [fluid, tank, rigid_body_master, rigid_body_slave,
                rigid_body_wall]

    def create_scheme(self):
        master = ParticlesFluidScheme(
            fluids=['fluid'],
            boundaries=['tank'],
            # rigid_bodies_combined=[],
            rigid_bodies_master=["rigid_body_combined_master"],
            rigid_bodies_slave=["rigid_body_combined_slave"],
            rigid_bodies_wall=["rigid_body_wall"],
            stirrer=[],
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

        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=200)
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
                if (t // self.T) % 2 == 0:
                    pa.u[:] = self.stirrer_velocity
                    pa.x[:] += pa.u[:] * dt
                else:
                    pa.u[:] = -self.stirrer_velocity
                    pa.x[:] += pa.u[:] * dt

    def customize_output(self):
        self._mayavi_config('''
        # b = particle_arrays['rigid_body']
        # b.scalar = 'm'
        b = particle_arrays['fluid']
        b.scalar = 'vmag'
        ''')

    def post_process(self, fname):

        import matplotlib.pyplot as plt
        from pysph.solver.utils import load, get_files
        from pysph.solver.utils import iter_output
        import os

        info = self.read_info(fname)
        files = self.output_files

        # initial position of the cylinder
        fname = files[0]
        data = load(fname)
        rigid_body = data['arrays']['rigid_body_combined_master']
        y_0 = rigid_body.y[0]

        y_current = []
        v_current = []
        u = []
        t_current = []
        fy = []

        for sd, rigid_body in iter_output(files, 'rigid_body_combined_master'):
            _t = sd['t']
            y_current.append(rigid_body.y[0])
            v_current.append(rigid_body.v[0])
            t_current.append(_t)
        # print(fy)
        # print(t, "t is ")
        # non dimentionalize it
        # penetration_current = (np.asarray(y)[::1] - y_0) / self.rigid_body_diameter
        # t_current = np.asarray(t)[::1] * (9.81 / self.rigid_body_diameter)**0.5

        # Data from literature
        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)

        # # load the data
        # # We use Sun 2018 accurate and efficient water entry paper data for validation
        # if self.rigid_body_rho == 500.:
        #     data_y_penetration_sun_2018_exp = np.loadtxt(os.path.join(
        #         directory, 'sun_2018_falling_500_rho_experimental_data.csv'), delimiter=',')
        #     data_y_penetration_sun_2018_BEM = np.loadtxt(os.path.join(
        #         directory, 'sun_2018_falling_500_rho_BEM_data.csv'), delimiter=',')
        #     # This is 200 resolution D / dx = 200
        #     data_y_penetration_sun_2018_SPH = np.loadtxt(os.path.join(
        #         directory, 'sun_2018_falling_500_rho_SPH_data.csv'), delimiter=',')
        # if self.rigid_body_rho == 1000.:
        #     data_y_penetration_sun_2018_exp = np.loadtxt(os.path.join(
        #         directory, 'sun_2018_falling_1000_rho_experimental_data.csv'), delimiter=',')
        #     data_y_penetration_sun_2018_BEM = np.loadtxt(os.path.join(
        #         directory, 'sun_2018_falling_1000_rho_BEM_data.csv'), delimiter=',')
        #     # This is 200 resolution D / dx = 200
        #     data_y_penetration_sun_2018_SPH = np.loadtxt(os.path.join(
        #         directory, 'sun_2018_falling_1000_rho_SPH_data.csv'), delimiter=',')

        # t_exp, penetration_exp = data_y_penetration_sun_2018_exp[:, 0], data_y_penetration_sun_2018_exp[:, 1]
        # t_BEM, penetration_BEM = data_y_penetration_sun_2018_BEM[:, 0], data_y_penetration_sun_2018_BEM[:, 1]
        # t_SPH, penetration_SPH = data_y_penetration_sun_2018_SPH[:, 0], data_y_penetration_sun_2018_SPH[:, 1]
        # # =================
        # # sort webplot data
        # p = t_SPH.argsort()
        # t_SPH = t_SPH[p]
        # penetration_SPH = penetration_SPH[p]
        # # t_SPH = np.delete(t_SPH, np.where(t_SPH > 1.65 and t_SPH < 1.75))
        # # penetration_SPH = np.delete(penetration_SPH,  np.where(t_SPH > 1.65 and t_SPH < 1.75))
        # t_SPH = np.delete(t_SPH, -4)
        # penetration_SPH = np.delete(penetration_SPH, -4)

        # p = t_BEM.argsort()
        # t_BEM = t_BEM[p]
        # penetration_BEM = penetration_BEM[p]
        # # sort webplot data
        # # =================

        res = os.path.join(self.output_dir, "results.npz")
        np.savez(res,
                 # t_exp=t_exp,
                 # penetration_exp=penetration_exp,
                 # t_BEM=t_BEM,
                 # penetration_BEM=penetration_BEM,
                 # t_SPH=t_SPH,
                 # penetration_SPH=penetration_SPH,
                 t_current=t_current,
                 y_current=y_current,
                 v_current=v_current)
        data = np.load(res)

        # ========================
        # Variation of y penetration
        # ========================
        plt.clf()
        # plt.plot(t_exp, penetration_exp, "^", label='Experimental')
        # plt.plot(t_SPH, penetration_SPH, "-+", label='SPH')
        # plt.plot(t_BEM, penetration_BEM, "--", label='BEM')
        # plt.plot(t_current, -penetration_current, "-", label='Current')
        # print("len of t is", len(t))
        # print("len of v is", len(fy))
        plt.plot(t_current, y_current)

        plt.title('Variation in y-position')
        plt.xlabel('t')
        plt.ylabel('Y-position')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "y_position_vs_t.png")
        plt.savefig(fig, dpi=300)
        # ========================
        # Variation of y penetration ends
        # ========================

        # ========================
        # Variation of v velocity
        # ========================
        plt.clf()
        # plt.plot(t_exp, penetration_exp, "^", label='Experimental')
        # plt.plot(t_SPH, penetration_SPH, "-+", label='SPH')
        # plt.plot(t_BEM, penetration_BEM, "--", label='BEM')
        # plt.plot(t_current, -penetration_current, "-", label='Current')
        # print("len of t is", len(t))
        # print("len of v is", len(fy))
        plt.plot(t_current, v_current)

        plt.title('Variation in v-velocity')
        plt.xlabel('t')
        plt.ylabel('v-velocity')
        plt.legend()
        fig = os.path.join(os.path.dirname(fname), "v_velocity_vs_t.png")
        plt.savefig(fig, dpi=300)
        # ========================
        # Variation of v velocity ends
        # ========================


if __name__ == '__main__':
    app = Problem()
    app.run()
    app.post_process(app.info_filename)
