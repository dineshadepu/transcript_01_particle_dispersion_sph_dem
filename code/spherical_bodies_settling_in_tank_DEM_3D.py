"""

Run it using:

python 04_3d_multiple_rigid_bodies_settling_in_tank.py --openmp

"""
import numpy as np
import sys

import pysph.tools.geometry as G
from pysph.base.utils import get_particle_array

from pysph.base.kernels import (QuinticSpline)
from pysph.solver.solver import Solver
from pysph.solver.application import Application
from pysph.tools.sph_evaluator import SPHEvaluator
from pysph.sph.scheme import SchemeChooser

from sph_dem.geometry import (get_fluid_tank_3d, get_truncated_circle_from_3d_block)

from sph_dem.rigid_body.rigid_body_3d import (RigidBody3DScheme,
                                              setup_rigid_body,
                                              set_linear_velocity,
                                              set_angular_velocity,
                                              get_master_and_slave_rb,
                                              add_contact_properties_body_master)

from pysph_dem.dem_simple import (setup_wall_dem)


class Problem(Application):
    def add_user_options(self, group):
        group.add_argument("--N", action="store", type=int, dest="N",
                           default=10,
                           help="Number of particles in diamter of a rigid cylinder")

        group.add_argument("--rigid-body-rho", action="store", type=float,
                           dest="rigid_body_rho", default=1500,
                           help="Density of rigid cylinder")

        group.add_argument("--rigid-body-diameter", action="store", type=float,
                           dest="rigid_body_diameter", default=1e-3,
                           help="Diameter of each particle")

        group.add_argument("--no-of-bodies", action="store", type=int,
                           dest="no_of_bodies", default=50,
                           help="Total no of rigid bodies")

        group.add_argument("--z-rb-layers", action="store", type=int,
                           dest="z_rb_layers", default=3,
                           help="No of layers of rigid bodies in z-direction")

    def consume_user_options(self):
        # ======================
        # Dimensions
        # ======================
        # tank geometry
        self.tank_layers = 0.
        # dimensions rigid body dimensions
        # All the particles are in circular or spherical shape
        self.rigid_body_diameter = 0.11
        self.rigid_body_velocity = 0.
        self.no_of_bodies = self.options.no_of_bodies
        if self.no_of_bodies % 5 != 0:
            print("no of bodies should be a multiple of 5")
            sys.exit()
        self.z_rb_layers = self.options.z_rb_layers
        # ======================
        # Dimensions ends
        # ======================

        # ======================
        # Physical properties and consants
        # ======================
        self.rigid_body_rho = self.options.rigid_body_rho
        self.rigid_body_E = 1e9
        self.rigid_body_nu = 0.23

        self.gx = 0.
        self.gy = -9.81
        self.gz = 0.
        self.dim = 3
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
        self.tf = 1.0

        # Setup default parameters.
        self.dt = 1e-4
        # ==========================
        # Numerical properties ends
        # ==========================

    def create_rb_geometry_particle_array(self):
        x = np.array([])
        y = np.array([])
        z = np.array([])
        cylinders_in_layer = 5
        for i in range(int(self.no_of_bodies / cylinders_in_layer)):
            j = 0
            x1, y1, z1 = get_truncated_circle_from_3d_block(
                self.rigid_body_diameter, self.dx)
            x1 = x1.ravel()
            y1 = y1.ravel()
            z1 = z1.ravel()
            x2 = np.array([])
            y2 = np.array([])
            z2 = np.array([])
            while j < cylinders_in_layer:
                # print(i, ": i is" )
                x1[:] = x1[:] + self.rigid_body_diameter + self.dx * 2
                x2 = np.concatenate((x2, x1[:]))
                y2 = np.concatenate((y2, y1[:]))
                z2 = np.concatenate((z2, z1[:]))
                j += 1
            y2[:] += i * self.rigid_body_diameter + self.dx * i
            x = np.concatenate((x, x2 + i % 2 * self.dx * 2.))
            y = np.concatenate((y, y2))
            z = np.concatenate((z, z2))

        x_all = np.array([])
        y_all = np.array([])
        z_all = np.array([])
        for i in range(self.z_rb_layers):
            x_all = np.concatenate((x_all, x))
            y_all = np.concatenate((y_all, y))
            z_all = np.concatenate((z_all, z + i * self.rigid_body_diameter + 2. * self.dx))

        x = x_all
        y = y_all
        z = z_all
        # y[:] += self.fluid_height + self.rigid_body_diameter
        # x[:] += self.fluid_length/2. + self.rigid_body_diameter
        # x[:] += self.fluid_length/2. - self.rigid_body_diameter * 4.
        x[:] += self.rigid_body_diameter * 4.

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
                                                 E=1e6,
                                                 nu=0.23,
                                                 rho=self.rigid_body_rho)
        body_id = np.array([])
        dem_id = np.array([])
        for i in range(self.no_of_bodies * self.z_rb_layers):
            body_id = np.concatenate((body_id, i * np.ones_like(x1, dtype='int')))
            dem_id = np.concatenate((dem_id, i * np.ones_like(x1, dtype='int')))

        rigid_body_combined.add_property('body_id', type='int', data=body_id)
        rigid_body_combined.add_property('dem_id', type='int', data=dem_id)
        return rigid_body_combined

    def create_particles(self):
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
        # move it to right, so that we can have a separate view
        disp_x = min(rigid_body_combined.x)
        rigid_body_combined.x[:] += disp_x

        # This is # 2, (Here we create a rigid body which is compatible with
        # combined rigid body solver formulation)
        setup_rigid_body(rigid_body_combined, self.dim, 5)
        rigid_body_combined.h[:] = self.h
        rigid_body_combined.rad_s[:] = self.dx / 2.

        # print("moi body ", body.inertia_tensor_inverse_global_frame)
        lin_vel = np.array([])
        ang_vel = np.array([])
        sign = -1
        for i in range(self.no_of_bodies * self.z_rb_layers):
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
        rigid_body_master.h[:] = self.rigid_body_diameter * 0.6
        add_contact_properties_body_master(rigid_body_master, 6, 5)

        # =========================
        # create rigid body ends
        # =========================

        # ======================
        # create wall for rigid body
        # ======================
        # left right bottom front back
        x = np.array([min(rigid_body_master.x) - self.rigid_body_diameter * 5.,
                      max(rigid_body_master.x) + self.rigid_body_diameter * 5.,
                      min(rigid_body_master.x) + (max(rigid_body_master.x) -
                                                  min(rigid_body_master.x)) / 2,
                      min(rigid_body_master.x) + (max(rigid_body_master.x) -
                                                  min(rigid_body_master.x)) / 2,
                      min(rigid_body_master.x) + (max(rigid_body_master.x) -
                                                  min(rigid_body_master.x)) / 2])
        # x[:] += disp_x
        y = np.array([self.rigid_body_diameter * 4.,
                      self.rigid_body_diameter * 4.,
                      min(rigid_body_master.y) - self.rigid_body_diameter * 1.,
                      self.rigid_body_diameter * 4.,
                      self.rigid_body_diameter * 4.,
                      ])
        z = np.array([0.,
                      0.,
                      0.,
                      max(rigid_body_master.z) + self.rigid_body_diameter * 5.,
                      min(rigid_body_master.z) - self.rigid_body_diameter * 5.])
        normal_x = np.array([1., -1., 0., 0., 0.])
        normal_y = np.array([0., 0., 1., 0., 0.])
        normal_z = np.array([0., 0., 0., -1., 1.])
        rigid_body_wall = get_particle_array(name='rigid_body_wall',
                                             x=x,
                                             y=y,
                                             z=z,
                                             normal_x=normal_x,
                                             normal_y=normal_y,
                                             normal_z=normal_z,
                                             h=self.rigid_body_diameter/2.,
                                             rho_b=self.rigid_body_rho,
                                             rad_s=self.rigid_body_diameter/2.,
                                             E=69. * 1e6,
                                             nu=0.3,
                                             G=69. * 1e5)
        dem_id = np.array([0, 0, 0, 0, 0])
        rigid_body_wall.add_property('dem_id', type='int', data=dem_id)
        rigid_body_wall.add_constant('no_wall', [5])
        setup_wall_dem(rigid_body_wall)

        return [rigid_body_master, rigid_body_slave, rigid_body_wall]

    def create_scheme(self):
        rb = RigidBody3DScheme(
            rigid_bodies_master=["rigid_body_combined_master"],
            rigid_bodies_slave=["rigid_body_combined_slave"],
            rigid_bodies_wall=["rigid_body_wall"],
            dim=3,
            gy=0.)

        s = SchemeChooser(default='rb', rb=rb)
        return s

    def configure_scheme(self):
        tf = self.tf
        scheme = self.scheme
        scheme.configure(
            dim=self.dim,
            gy=self.gy)

        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=200)
        print("dt = %g"%self.dt)


if __name__ == '__main__':
    app = Problem()
    app.run()
    app.post_process(app.info_filename)
