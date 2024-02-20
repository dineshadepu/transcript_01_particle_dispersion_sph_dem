"""Change this

"""

import numpy as np
import matplotlib.pyplot as plt

from pysph.base.kernels import CubicSpline
# from pysph.base.utils import get_particle_array_rigid_body
from pysph.base.utils import get_particle_array
from pysph.sph.integrator import EPECIntegrator
from pysph.solver.application import Application
from pysph.sph.scheme import SchemeChooser

from sph_dem.geometry import (get_2d_block)

from sph_dem.rigid_body.rigid_body_3d import (setup_rigid_body,
                                              set_linear_velocity,
                                              set_angular_velocity,
                                              get_master_and_slave_rb,
                                              add_contact_properties_body_master,
                                              RigidBody3DScheme)

from pysph_dem.dem_simple import (setup_wall_dem)


class Case0(Application):
    def add_user_options(self, group):
        # group.add_argument(
        #     "--re", action="store", type=float, dest="re", default=0.0125,
        #     help="Reynolds number of flow."
        # )

        group.add_argument("--no-of-bodies", action="store", type=int, dest="no_of_bodies",
                           default=10,
                           help="Number of freely moving bodies")

        # group.add_argument(
        #     "--remesh", action="store", type=float, dest="remesh", default=0,
        #     help="Remeshing frequency (setting it to zero disables it)."
        # )

    def consume_user_options(self):
        # ======================
        # Get the user options and save them
        # ======================
        self.no_of_bodies = self.options.no_of_bodies
        # ======================
        # ======================

        self.rho0 = 2700.0
        self.hdx = 1.0
        self.dy = 0.1
        self.kn = 1e4
        self.mu = 0.5
        self.en = 1.0
        self.E = 69. * 1e5
        self.dim = 2
        self.body_diameter = 1.
        self.dx = self.body_diameter / 20
        self.gy = -9.81
        # self.gy = 0.

        # x - axis
        self.stirrer_length = self.body_diameter
        # y - axis
        self.stirrer_height = 5. * self.body_diameter
        # z - axis
        self.stirrer_depth = 4. * self.body_diameter
        self.stirrer_velocity = 5.
        # time period of stirrer
        self.T = 6. * self.body_diameter / self.stirrer_velocity

        self.dt = 5e-4
        self.tf = 3.

    def create_stirrer(self):
        dx = self.dx
        x_stirrer, y_stirrer = get_2d_block(dx=self.dx,
                                            length=self.stirrer_length,
                                            height=self.stirrer_height)
        m = np.ones_like(x_stirrer) * dx * dx * self.rho0
        h = np.ones_like(x_stirrer) * self.hdx * dx
        stirrer_dem = get_particle_array(name='stirrer_dem',
                                         x=x_stirrer, y=y_stirrer,
                                         u=self.stirrer_velocity,
                                         h=h, m=m,
                                         rho=self.rho0,
                                         E=self.E,
                                         nu=0.3,
                                         G=self.E,
                                         rad_s=self.dx)
        dem_id = np.ones_like(x_stirrer, dtype='int')
        stirrer_dem.add_property('dem_id', type='int', data=dem_id)
        return stirrer_dem

    def create_particles(self):
        from pysph_dem.geometry import create_circle_1
        dx = self.dx
        x1, y1 = create_circle_1(self.body_diameter, dx)
        x1 = x1.ravel()
        y1 = y1.ravel()
        x = np.array([])
        y = np.array([])
        sign = -1
        for i in range(self.no_of_bodies):
            sign *= -1
            x = np.concatenate((x, x1[:] + i * self.body_diameter + i * 2. * dx))
            y = np.concatenate((y, y1[:] + sign * self.body_diameter / 4.))

        x2 = np.array([])
        y2 = np.array([])
        for i in range(self.no_of_bodies):
            sign *= -1
            x2 = np.concatenate((x2, x1[:] + i * self.body_diameter + i * 2. * dx))
            y2 = np.concatenate((y2, y1[:] + sign * self.body_diameter / 4.))
        y2 += self.body_diameter * 3.

        x3 = np.array([])
        y3 = np.array([])
        for i in range(self.no_of_bodies):
            sign *= -1
            x3 = np.concatenate((x3, x1[:] + i * self.body_diameter + i * 2. * dx))
            y3 = np.concatenate((y3, y1[:] + sign * self.body_diameter / 4.))
        y3 += self.body_diameter * 6.

        x = np.concatenate((x, x2, x3))
        y = np.concatenate((y, y2, y3))

        self.no_of_bodies = 3 * self.no_of_bodies

        m = np.ones_like(x) * dx * dx * self.rho0
        h = np.ones_like(x) * self.hdx * dx
        # radius of each sphere constituting in cube
        rad_s = np.ones_like(x) * dx
        body = get_particle_array(name='body', x=x, y=y, h=h, m=m,
                                  rho=self.rho0,
                                  rad_s=rad_s,
                                  E=self.E,
                                  nu=0.3)
        body_id = np.array([])
        dem_id = np.array([])
        for i in range(self.no_of_bodies):
            body_id = np.concatenate((body_id, i * np.ones_like(x1, dtype='int')))
            dem_id = np.concatenate((dem_id, i * np.ones_like(x1, dtype='int')))
        # print(len(body.x))
        # print(len(body_id))
        body.add_property('body_id', type='int', data=body_id)
        body.add_property('dem_id', type='int', data=dem_id)

        # setup the properties
        setup_rigid_body(body, self.dim)

        # print("moi body ", body.inertia_tensor_inverse_global_frame)
        lin_vel = np.array([])
        ang_vel = np.array([])
        sign = -0
        for i in range(self.no_of_bodies):
            sign *= -1
            lin_vel = np.concatenate((lin_vel, np.array([sign * 5., 0., 0.])))
            ang_vel = np.concatenate((ang_vel, np.array([0., 0., 0.])))

        set_linear_velocity(body, lin_vel)
        set_angular_velocity(body, ang_vel)

        body_master, body_slave = get_master_and_slave_rb(body)
        add_contact_properties_body_master(body_master, 6, 2)
        body_master.rad_s[:] = self.body_diameter / 2.
        body_master.h[:] = self.body_diameter * 2.

        # ======================
        # create wall
        # ======================
        x = np.array([min(body_slave.x) - self.dx * 8, max(body_slave.x) + self.dx * 8,
                      body_master.x[1], body_master.x[1]])
        y = np.array([body_master.y[0], body_master.y[0],
                      body_master.y[1] - self.body_diameter * 3.,
                      body_master.y[1] + self.body_diameter * 10.,
                      ])
        normal_x = np.array([1., -1., 0., 0.])
        normal_y = np.array([0., 0., 1., -1])
        normal_z = np.array([0., 0., 0., 0.])
        wall = get_particle_array(name='wall',
                                  x=x,
                                  y=y,
                                  normal_x=normal_x,
                                  normal_y=normal_y,
                                  normal_z=normal_z,
                                  h=self.body_diameter / 4.,
                                  rho_b=self.rho0,
                                  rad_s=self.body_diameter / 2.,
                                  E=69. * 1e5,
                                  nu=0.3,
                                  G=69. * 1e5,
                                  )
        dem_id = np.array([0, 0, 0, 0])
        wall.add_property('dem_id', type='int', data=dem_id)
        wall.add_constant('no_wall', [4])
        setup_wall_dem(wall)

        # ======================
        # create stirrer
        # ======================
        stirrer_dem = self.create_stirrer()
        stirrer_dem.x[:] -= self.body_diameter * 2.
        # stirrer_dem.x[:] = self.body_diameter * 20.
        stirrer_dem.y[:] -= min(stirrer_dem.y) - min(wall.y)
        stirrer_dem.u[:] = self.stirrer_velocity

        return [body_master, body_slave, wall, stirrer_dem]

    def create_scheme(self):
        rb3d_ms = RigidBody3DScheme(rigid_bodies_master=['body_master'],
                                    rigid_bodies_slave=['body_slave'],
                                    rigid_bodies_wall=['wall'],
                                    stirrer=["stirrer_dem"],
                                    dim=2,
                                    gy=0.)

        s = SchemeChooser(default='rb3d_ms', rb3d_ms=rb3d_ms)

        return s

    def configure_scheme(self):
        tf = self.tf
        scheme = self.scheme
        scheme.configure(
            gy=self.gy,
            dim=self.dim)

        scheme.configure_solver(tf=tf, dt=self.dt, pfreq=100)
        print("dt = %g"%self.dt)

    def post_step(self, solver):
        t = solver.t
        dt = solver.dt
        for pa in self.particles:
            if pa.name == 'stirrer_dem':
                if (t // self.T) % 2 == 0:
                    pa.u[:] = self.stirrer_velocity
                    pa.x[:] += pa.u[:] * dt
                else:
                    pa.u[:] = -self.stirrer_velocity
                    pa.x[:] += pa.u[:] * dt

    def post_process(self, fname):
        if len(self.output_files) == 0:
            return

        from pysph.solver.utils import iter_output

        files = self.output_files
        files = files[:]
        t, total_energy = [], []
        x, y = [], []
        R = []
        ang_mom = []
        for sd, body in iter_output(files, 'body_master'):
            _t = sd['t']
            # print(_t)
            t.append(_t)
            total_energy.append(0.5 * np.sum(body.m[:] * (body.u[:]**2. +
                                                          body.v[:]**2.)))
            R.append(body.R[0])
            # print("========================")
            # print("R is", body.R)
            # # print("ang_mom x is", body.ang_mom_x)
            # # print("ang_mom y is", body.ang_mom_y)
            # print("ang_mom z is", body.ang_mom_z)
            # # print("omega x is", body.omega_x)
            # # print("omega y is", body.omega_y)
            # print("omega z is", body.omega_z)
            # print("moi global master ", body.inertia_tensor_inverse_global_frame)
            # # print("moi body master ", body.inertia_tensor_inverse_body_frame)
            # # print("moi global master ", body.inertia_tensor_global_frame)
            # # print("moi body master ", body.inertia_tensor_body_frame)
            # # x.append(body.xcm[0])
            # # y.append(body.xcm[1])
            # # print(body.ang_mom_z[0])
            ang_mom.append(body.ang_mom_z[0])
        # print(ang_mom)

        import matplotlib
        import os
        # matplotlib.use('Agg')

        from matplotlib import pyplot as plt

        # res = os.path.join(self.output_dir, "results.npz")
        # np.savez(res, t=t, amplitude=amplitude)

        # gtvf data
        # data = np.loadtxt('./oscillating_plate.csv', delimiter=',')
        # t_gtvf, amplitude_gtvf = data[:, 0], data[:, 1]

        plt.clf()

        # plt.plot(t_gtvf, amplitude_gtvf, "s-", label='GTVF Paper')
        # plt.plot(t, total_energy, "-", label='Simulated')
        # plt.plot(t, ang_mom, "-", label='Angular momentum')
        plt.plot(t, R, "-", label='R[0]')

        plt.xlabel('t')
        plt.ylabel('ang energy')
        plt.legend()
        fig = os.path.join(self.output_dir, "ang_mom_vs_t.png")
        plt.savefig(fig, dpi=300)
        # plt.show()

        # plt.plot(x, y, label='Simulated')
        # plt.show()


if __name__ == '__main__':
    app = Case0()
    app.run()
    # app.post_process(app.info_filename)
