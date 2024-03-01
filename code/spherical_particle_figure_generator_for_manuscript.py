import numpy as np
import matplotlib.pyplot as plt

from pysph_dem.geometry import create_circle_1
from pysph_rfc_new.geometry import hydrostatic_tank_2d, create_circle_1, translate_system_with_left_corner_as_origin
from pysph.base.utils import get_particle_array
import pysph.tools.geometry as G


rigid_body_diameter = 0.3
rigid_body_radius = 0.3 / 2.
dx = rigid_body_diameter / 10
fluid_length_ratio = 5.
fluid_height_ratio = 5.
tank_length_ratio = 1.2
tank_height_ratio = 1.2


# x - axis
fluid_length = fluid_length_ratio * rigid_body_diameter
# y - axis
fluid_height = fluid_height_ratio * rigid_body_diameter
# z - axis
fluid_depth = 0.

# x - axis
tank_length = tank_length_ratio * fluid_length
# y - axis
tank_height = tank_height_ratio * fluid_height
# z - axis
tank_depth = 0.0

tank_layers = 3


def hs_tank_with_spherical_particles():
    xf, yf, xt, yt = hydrostatic_tank_2d(fluid_length, fluid_height,
                                         tank_height, tank_layers,
                                         dx, dx, False)

    fluid = get_particle_array(name='rb',
                               x=xf,
                               y=yf,
                               h=1.2 * dx)
    tank = get_particle_array(name='rb',
                              x=xt,
                              y=yt,
                              h=1.2 * dx)

    x1, y1 = create_circle_1(rigid_body_diameter, dx)

    x2, y2 = create_circle_1(rigid_body_diameter * 1.2, dx)
    x2 += rigid_body_diameter
    y2 -= rigid_body_diameter
    x2 -= rigid_body_radius * 0.4
    y2 += rigid_body_radius

    x3, y3 = create_circle_1(rigid_body_diameter, dx)
    x3 -= 1.5 * rigid_body_diameter
    y3 += max(xf) - min(y3) - rigid_body_radius

    x4, y4 = create_circle_1(rigid_body_diameter, dx)
    x4 -= 1.5 * rigid_body_diameter
    y4 -= max(xf) - min(y3) + rigid_body_diameter
    x = np.concatenate((x1, x2, x3, x4))
    y = np.concatenate((y1, y2, y3, y4))
    rb = get_particle_array(name='rb',
                            x=x,
                            y=y,
                            h=1.2 * dx)
    # remove fluid particles overlapping with the rigid body
    G.remove_overlap_particles(
        fluid, rb, dx, dim=2
    )

    plt.scatter(fluid.x, fluid.y, s=1)
    plt.scatter(tank.x, tank.y, s=1)

    plt.scatter(rb.x, rb.y, s=1)

    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.savefig('hs_tank_with_spherical_particles.png', dpi=300)
    # plt.show()


hs_tank_with_spherical_particles()


# body_diameter = 1.
# dx = 0.05
# x1, y1 = create_circle_1(body_diameter, dx)
# plt.scatter(x1, y1)
# plt.gca().set_aspect('equal')
# plt.axis('off')
# # plt.spines['top'].set_visible(False)
# # plt.spines['right'].set_visible(False)
# # plt.spines['bottom'].set_visible(False)
# # plt.spines['left'].set_visible(False)
# plt.savefig('cirlce.png')
# # plt.show()

# plt.clf()

# rad = 0.2
# overlap = 0.2 / 10
# x_centers = np.array([1.5 * rad, 1.5 * rad + 2. * rad - overlap])
# y_centers = np.array([2. * rad, 2. * rad])
# circles = []
# for i in range(len(x_centers)):
#     circles.append(plt.Circle((x_centers[i], y_centers[i]), rad, color='b', fill=False))

# # circle2 = plt.Circle((0.5, 0.5), 0.2, color='blue')
# # circle3 = plt.Circle((1, 1), 0.2, color='g', clip_on=False)

# fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
# # (or if you have an existing figure)
# # fig = plt.gcf()
# # ax = fig.gca()

# for i in range(len(x_centers)):
#     ax.add_patch(circles[i])


# plt.gca().set_aspect('equal')
# fig.savefig('plotcircles.png')
# # plt.show()
