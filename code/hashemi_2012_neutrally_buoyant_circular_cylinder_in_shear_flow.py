"""A neutrally buoyand circular cylinder in a shear flow

Section 7.1 in Hashemi 2012 paper

python hashemi_2012_neutrally_buoyant_circular_cylinder_in_shear_flow.py --openmp --no-use-edac --nu 1e-6 --alpha 0.02 --artificial-viscosity-with-boundary --pfreq 300 --internal-flow --pst

"""
import os

# numpy
import numpy as np

# PySPH imports
from pysph.base.nnps import DomainManager
from pysph.base.utils import get_particle_array
from pysph.solver.application import Application
from pysph.solver.utils import load
import pysph.tools.geometry as G
import sys
sys.path.insert(0, "./../")
from pysph_rfc_new.fluids import (get_particle_array_fluid, get_particle_array_boundary)
from pysph.sph.scheme import SchemeChooser
from pysph.sph.equation import Group, MultiStageEquations

from pysph.sph.equation import Equation, Group
from pysph_rfc_new.geometry import hydrostatic_tank_2d, create_circle_1, translate_system_with_left_corner_as_origin

from sph_dem.rigid_body.rigid_body_3d import (setup_rigid_body,
                                              set_linear_velocity,
                                              set_angular_velocity,
                                              get_master_and_slave_rb,
                                              add_contact_properties_body_master)

from sph_dem.rigid_fluid_coupling import (
    ParticlesFluidScheme, add_rigid_fluid_properties_to_rigid_body)
from pysph_dem.dem_simple import (setup_wall_dem)


def check_time_make_zero(t, dt):
    if t < 5.0:
        return True
    else:
        return False


def get_center_of_mass(x, y, z, m):
    # loop over all the bodies
    xcm = [0., 0., 0.]
    total_mass = np.sum(m)
    xcm[0] = np.sum(m[:] * x[:]) / total_mass
    xcm[1] = np.sum(m[:] * y[:]) / total_mass
    xcm[2] = np.sum(m[:] * z[:]) / total_mass
    return xcm


def move_body_to_new_center(xcm, x, y, z, center):
    x_trans = center[0] - xcm[0]
    y_trans = center[1] - xcm[1]
    z_trans = center[2] - xcm[2]
    x[:] += x_trans
    y[:] += y_trans
    z[:] += z_trans


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


class AdjustRigidBodyPositionInPipe(Equation):
    def __init__(self, dest, sources,
                 x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max
        super(AdjustRigidBodyPositionInPipe, self).__init__(dest, sources)

    def reduce(self, dst, t, dt):
        # frc = declare('object')
        # trq = declare('object')
        # fx = declare('object')
        # fy = declare('object')
        # fz = declare('object')
        x = declare('object')
        # y = declare('object')
        # z = declare('object')
        # dx0 = declare('object')
        # dy0 = declare('object')
        # dz0 = declare('object')
        # xcm = declare('object')
        # R = declare('object')
        # total_mass = declare('object')
        # body_id = declare('object')
        # nb = declare('object')
        # j = declare('int')
        i = declare('int')
        # i3 = declare('int')
        # i9 = declare('int')

        # frc = dst.force
        # trq = dst.torque
        # fx = dst.fx
        # fy = dst.fy
        # fz = dst.fz
        # nb = dst.nb
        x = dst.x
        # y = dst.y
        # z = dst.z
        # dx0 = dst.dx0
        # dy0 = dst.dy0
        # dz0 = dst.dz0
        # xcm = dst.xcm
        # R = dst.R
        # total_mass = dst.total_mass
        # body_id = dst.body_id

        for i in range(len(x)):
            # i = body_id[j]
            # i3 = 3 * i
            # i9 = 9 * i
            if x[i] > self.x_max:
                # recompute the center of mass based on the periodic particles
                # of the rigid body
                x[i] = self.x_min + (x[i] - self.x_max)
            elif x[i] < self.x_min:
                x[i] = self.x_max - (self.x_min - x[i])


class PoiseuilleFlow(Application):
    def add_user_options(self, group):
        group.add_argument(
            "--re", action="store", type=float, dest="re", default=0.625,
            help="Reynolds number of flow."
        )
        group.add_argument(
            "--remesh", action="store", type=float, dest="remesh", default=0,
            help="Remeshing frequency (setting it to zero disables it)."
        )

    def consume_user_options(self):
        # ======================
        # Get the user options and save them
        # ======================
        self.re = self.options.re
        # ======================
        # ======================

        # ======================
        # Dimensions
        # ======================
        H = 0.01
        L = 5 * H
        # x - axis
        self.fluid_length = L
        # y - axis
        self.fluid_height = H
        # z - axis
        self.fluid_depth = 0.0

        # x - axis
        self.tank_length = 0.0
        # y - axis
        self.tank_height = 0.0
        # z - axis
        self.tank_depth = 0.0
        self.tank_layers = 4

        R = 1. / 8. * H
        self.rigid_body_diameter = R * 2.
        # ======================
        # Dimensions ends
        # ======================

        # ======================
        # Velocity
        # ======================
        self.Umax = 0.02
        # ======================
        # Dimensions ends
        # ======================

        # ======================
        # Physical properties and consants
        # ======================
        self.fluid_rho = 1000.
        self.rigid_body_rho = 1000

        self.gx = 0.
        self.gy = 0.
        self.gz = 0.
        self.dim = 2
        # ======================
        # Physical properties and consants ends
        # ======================

        # ======================
        # Numerical properties
        # ======================
        self.hdx = 1.
        # self.dx = 1.0/60.0
        self.dx = self.rigid_body_diameter / 10
        self.h = self.hdx * self.dx
        self.vref = self.Umax
        self.c0 = 10.0 * self.Umax
        self.mach_no = self.vref / self.c0
        # set the viscosity based on the particle reynolds no
        self.nu = self.Umax * self.rigid_body_diameter**2. / (self.fluid_height * self.re)
        self.mu = self.nu * self.fluid_rho
        print("Kinematic viscosity is: ", self.nu)
        self.tf = 83.
        self.p0 = self.fluid_rho*self.c0**2
        self.alpha = 0.02

        # Setup default parameters.
        # dt_cfl = 0.25 * self.h / (self.c0 + self.vref)
        dt_viscous = 0.125 * self.h**2/self.nu
        # dt_force = 0.25 * np.sqrt(self.h/(self.gy))
        dt_force = 0.25 * np.sqrt(self.h/10)
        print("dt viscous is: ", dt_viscous)
        print("dt force is: ", dt_force)
        self.dt = min(dt_viscous, dt_force)
        # self.dt = 1e-4
        # self.dt = dt_viscous
        print("dt is: ", self.dt)

        # ==========================
        # Numerical properties ends
        # ==========================

        # ====================================================
        # Start: properties to be used while adjusting the equations
        # ====================================================
        Lx = self.fluid_length
        Ly = self.fluid_height
        _x = np.arange(self.dx/2, self.fluid_length, self.dx)

        # create the fluid particles
        _y = np.arange(self.dx/2, self.fluid_height, self.dx)

        x, y = np.meshgrid(_x, _y); fx = x.ravel(); fy = y.ravel()
        # self.x_min = min(fx)
        # self.x_max = max(fx)
        self.x_min = 0.0
        self.x_max = self.fluid_length
        print("x min is ", self.x_min)
        print("x max is ", self.x_max)
        # ====================================================
        # end: properties to be used while adjusting the equations
        # ====================================================

    def create_particles(self):
        Lx = self.fluid_length
        Ly = self.fluid_height
        _x = np.arange(self.dx/2, self.fluid_length, self.dx)

        # create the fluid particles
        _y = np.arange(self.dx/2, self.fluid_height, self.dx)

        x, y = np.meshgrid(_x, _y); fx = x.ravel(); fy = y.ravel()

        # create the channel particles at the top
        _y = np.arange(self.fluid_height+self.dx/2,
                       self.fluid_height+self.dx/2+self.tank_layers*self.dx,
                       self.dx)
        x, y = np.meshgrid(_x, _y); tx = x.ravel(); ty = y.ravel()

        # create the channel particles at the bottom
        _y = np.arange(-self.dx/2, -self.dx/2-self.tank_layers*self.dx, -self.dx)
        x, y = np.meshgrid(_x, _y); bx = x.ravel(); by = y.ravel()

        # concatenate the top and bottom arrays
        cx = np.concatenate((tx, bx))
        cy = np.concatenate((ty, by))

        # create the arrays
        channel = get_particle_array_boundary(name='channel', x=cx, y=cy)
        fluid = get_particle_array_fluid(name='fluid', x=fx, y=fy)

        # set velocities of the top particles of the channel
        channel.u[channel.y > self.fluid_height / 2.] = self.Umax / 2.
        channel.u[channel.y < self.fluid_height / 2.] = - self.Umax / 2.

        print("Poiseuille flow :: Re = %g, nfluid = %d, nchannel=%d"%(
            self.re, fluid.get_number_of_particles(),
            channel.get_number_of_particles()))

        # add requisite properties to the arrays:
        # self.scheme.setup_properties([fluid, channel])

        # setup the particle properties
        volume = self.dx * self.dx

        # mass is set to get the reference density of rho0
        fluid.m[:] = volume * self.fluid_rho
        channel.m[:] = volume * self.fluid_rho

        # Set the default rho.
        fluid.rho[:] = self.fluid_rho
        channel.rho[:] = self.fluid_rho

        # # volume is set as dx^2
        # fluid.V[:] = 1./volume
        # channel.V[:] = 1./volume

        # smoothing lengths
        fluid.h[:] = self.h
        channel.h[:] = self.h

        # Set the constants
        fluid.c0_ref[0] = self.c0
        fluid.p0_ref[0] = self.p0

        # =========================
        # create rigid body
        # =========================
        x, y = create_circle_1(self.rigid_body_diameter, self.dx)
        z = np.zeros_like(x)

        m = self.rigid_body_rho * self.dx**self.dim * np.ones_like(x)
        h = self.h

        center = [self.fluid_length/3., 0.75 * self.fluid_height, 0.]
        center = [self.rigid_body_diameter, 0.75 * self.fluid_height, 0.]
        center = [self.fluid_length - 10. * self.dx, 0.75 * self.fluid_height, 0.]
        center = [self.rigid_body_diameter, 2. * self.fluid_height, 0.]
        xcm = get_center_of_mass(x, y, z, m)
        move_body_to_new_center(xcm, x, y, z, center)
        x += self.fluid_length / 4.
        x += self.fluid_length / 2.
        y -= self.fluid_height
        y -= self.rigid_body_diameter

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

        body_id = np.array([], dtype=int)
        for i in range(1):
            b_id = np.ones(len(x), dtype=int) * i
            body_id = np.concatenate((body_id, b_id))

        print("body_id is", body_id)
        dem_id = body_id

        rigid_body_combined.add_property('body_id', type='int', data=body_id)
        rigid_body_combined.add_property('dem_id', type='int', data=dem_id)

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
        print("rigid body master x is", rigid_body_master.x)

        # This is # 4
        rigid_body_master.rad_s[:] = self.rigid_body_diameter / 2.
        # rigid_body_master.h[:] = self.rigid_body_diameter * 2.
        # changing this just for this current example
        rigid_body_master.h[:] = self.h
        add_contact_properties_body_master(rigid_body_master, 6, 2)

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

        # print("rigid body total mass: ", rigid_body.total_mass)
        rigid_body_combined.rho[:] = self.fluid_rho
        G.remove_overlap_particles(
            fluid, rigid_body_combined, self.dx, dim=self.dim
        )
        # =========================
        # create rigid body ends
        # =========================

        # ======================
        # create wall for rigid body
        # ======================
        # bottom top
        x = np.array([min(fluid.x) + self.fluid_length / 2.,
                      min(fluid.x) + self.fluid_length / 2.])
        y = np.array([min(channel.y) + self.tank_layers * self.dx,
                      max(channel.y) - self.tank_layers * self.dx])
        normal_x = np.array([0., 0.])
        normal_y = np.array([1., -1.])
        normal_z = np.array([0., 0.])
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
        dem_id = np.array([0, 0])
        rigid_body_wall.add_property('dem_id', type='int', data=dem_id)
        rigid_body_wall.add_constant('no_wall', [2])
        setup_wall_dem(rigid_body_wall)
        # ==================================
        # create wall for rigid body ends
        # ==================================
        # return the particle list
        print("slave particles of rigid body are", len(rigid_body_slave.x))
        return [fluid, channel, rigid_body_master, rigid_body_slave, rigid_body_wall]

    def create_domain(self):
        return DomainManager(xmin=0, xmax=self.fluid_length, periodic_in_x=True)

    def create_scheme(self):
        wcsph = ParticlesFluidScheme(
            fluids=['fluid'],
            boundaries=['channel'],
            rigid_bodies_master=["rigid_body_combined_master"],
            rigid_bodies_slave=["rigid_body_combined_slave"],
            rigid_bodies_wall=["rigid_body_wall"],
            stirrer=[],
            dim=0.,
            rho0=0.,
            h=0.,
            c0=0.,
            pb=0.,
            nu=0.,
            gy=0.,
            alpha=0.)
        s = SchemeChooser(default='wcsph', wcsph=wcsph)
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
            gy=self.gy,
            alpha=self.alpha)

        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=1000)

    def create_equations(self):
        # print("inside equations")
        eqns = self.scheme.get_equations()

        # Apply external force
        adjust_eqs = []
        adjust_eqs.append(
            AdjustRigidBodyPositionInPipe(
                "rigid_body_combined_master", sources=None, x_min=self.x_min, x_max=self.x_max))

        adjust_eqs.append(
            AdjustRigidBodyPositionInPipe(
                "rigid_body_combined_slave", sources=None, x_min=self.x_min, x_max=self.x_max))

        eqns.groups[0].append(Group(adjust_eqs))

        # eqns.groups[1].append(Group(adjust_eqs))

        # # Apply external force
        # zero_frc = []
        # zero_frc.append(
        #     MakeForcesZeroOnRigidBody("rigid_body", sources=None))

        # # print(eqns.groups)
        # eqns.groups[-1].append(Group(equations=zero_frc,
        #                              condition=check_time_make_zero))

        return eqns

    def create_tools(self):
        tools = []
        if self.options.remesh > 0:
            from pysph.solver.tools import SimpleRemesher
            remesher = SimpleRemesher(
                self, 'fluid', props=['u', 'v', 'uhat', 'vhat'],
                freq=self.options.remesh
            )
            tools.append(remesher)
        return tools

    def post_process(self, fname):
        import matplotlib.pyplot as plt
        from pysph.solver.utils import load, get_files

        output_files = get_files(os.path.dirname(fname))

        from pysph.solver.utils import iter_output

        files = output_files

        t_current = []

        vertical_position_current = []
        u_cm = []

        step = 1
        for sd, rigid_body in iter_output(files[::step], 'rigid_body'):
            _t = sd['t']
            if _t > 3:
                print(_t)

                vertical_position_current.append(rigid_body.xcm[1])
                u_cm.append(rigid_body.vcm[0])
                t_current.append(_t)
        t_current = np.asarray(t_current)
        t_current -= 3.
        vertical_position_current = np.asarray(vertical_position_current)

        # Data from literature
        path = os.path.abspath(__file__)
        directory = os.path.dirname(path)

        # load the data
        # We use Sun 2018 accurate and efficient water entry paper data for validation
        data_vertical_postion_feng_2018_exp = np.loadtxt(os.path.join(
            directory, 'hashemi_2012_neutrally_inertial_migration_Z_G_Feng_2002_vertical_position_data.csv'), delimiter=',')

        t_feng, vertical_position_feng = data_vertical_postion_feng_2018_exp[:, 0], data_vertical_postion_feng_2018_exp[:, 1]

        res = os.path.join(self.output_dir, "results.npz")
        np.savez(res,
                 t_feng, vertical_position_feng,
                 t_current, vertical_position_current)

        plt.plot(t_current, vertical_position_current, label="Current SPH")
        plt.plot(t_feng, vertical_position_feng, label="Feng 2002")
        plt.ylabel("Vertical position (m)")
        plt.xlabel("Time (seconds)")
        plt.legend()
        fig = os.path.join(self.output_dir, "t_vs_y_cm.png")
        plt.savefig(fig, dpi=300)
        plt.clf()


if __name__ == '__main__':
    app = PoiseuilleFlow()
    app.run()
    app.post_process(app.info_filename)
