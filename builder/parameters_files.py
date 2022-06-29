"""This module has functions that creates some parameters files"""

import numpy as np
import matplotlib.pyplot as plt


def integra(vv):
    return np.sum(vv[1:-1]) + (vv[0] + vv[-1]) / 2.0


def create_velocity_file(
    model_name: str,
    vel_array: np.array,
    phi_array: np.array,
    Lx: int or float,
    Ly: int or float,
    X: np.array,
    Y: np.array,
    Nx: int,
    Ny: int,
):
    """
    Create the 2D grid in which the interfaces will be plotted.

    Parameters
    ----------------------------------------------------------------------------
    model_name: str
        Name of the model
    vel_array: np.array
        Array containing the velocity
    phi_array: np.array
        Array containing the velocity angles
    x_size: int or float
        Horizontal axis length in meters.
        Must be a positive and non-zero value.
    y_size: int or float
        Vertical axis length in meters.
        Must be a positive and non-zero value.
    az_width: int or float
        Thickness of the accommodation zone in meters.
        Must be a positive and non-zero value.
    Nx: int
        Number of nodes in the horizontal direction (x-axis).
        Must be a positive and non-zero value.
    Ny: int
        Number of nodes in the vertical direction (y-axis).
        Must be a positive and non-zero value.

    Return
    ----------------------------------------------------------------------------
    x, y: np.array
        1D coordinate array along the x- and y-axis
    X, Y: np.array
        2D coordinate grids along the x- and y-axis.
    Lx, Ly: int or float
        Total horizontal and vertical extents of the model in meters.
    """
    cont_v = 0

    for vel, phi in zip(vel_array, phi_array):

        vL = vel / 3.154e10  # m/s
        vLy = vL * Ly / Lx
        vLx = vL

        XX = X - (np.max(X) + np.min(X)) / 2.0 + 1.0e-5
        YY = Y - (np.max(Y) + np.min(Y)) / 2.0 + 1.0e-5

        r0 = 600.0e3
        XX = XX * vL / r0
        YY = YY * vL / r0

        theta = 2 * phi * np.pi / 180.0

        Vx = (-XX) * np.cos(theta) + YY * np.sin(theta)
        Vy = (XX) * np.sin(theta) + YY * np.cos(theta)

        Ux = vL
        Uy = vL

        pi = np.pi

        A = 0.0
        B = (2 * pi * Uy - pi * pi * Ux) / (pi * pi - 4)
        C = (4 * Ux - 2 * pi * Uy) / (pi * pi - 4)
        D = (2 * pi * Ux - 4 * Uy) / (pi * pi - 4)

        XN = np.abs(XX)
        YN = np.abs(YY)

        VVx = np.sign(XX) * (
            -B
            - D * (np.arctan2(YN, XN))
            + (C * XN + D * YN) * (-XN / (XX**2 + YY**2))
        )
        VVy = np.sign(YY) * (
            A
            + C * (np.arctan2(YN, XN))
            + (C * XN + D * YN) * (-YN / (XX**2 + YY**2))
        )

        Vx = (VVx) * np.cos(theta) + VVy * np.sin(theta)
        Vy = (-VVx) * np.sin(theta) + VVy * np.cos(theta)

        va1x = Vx[X == np.min(X)]
        va2x = Vx[X == np.max(X)]

        va1y = Vy[Y == np.min(Y)]
        va2y = Vy[Y == np.max(Y)]

        flowx = integra(va1x - va2x) * Ly / (Ny - 1)
        flowy = integra(va1y - va2y) * Lx / (Nx - 1)
        flow = flowx + flowy

        for i in range(600):
            Vx[X == np.max(X)] += flow / Ly
            va2x = Vx[X == np.max(X)]

            flowx = integra(va1x - va2x) * Ly / (Ny - 1)
            flowy = integra(va1y - va2y) * Lx / (Nx - 1)
            flow = flowx + flowy

        flowx = integra(va1x - va2x) * Ly / (Ny - 1)
        flowy = integra(va1y - va2y) * Lx / (Nx - 1)
        flow = flowx + flowy

        Vx[1:-1, 1:-1] = 0
        Vy[1:-1, 1:-1] = 0

        Vvx = np.copy(np.reshape(Vx, Nx * Ny))
        Vvy = np.copy(np.reshape(Vy, Nx * Ny))

        v = np.zeros((2, Nx * Ny))

        v[0, :] = Vvx
        v[1, :] = Vvy

        v = np.reshape(v.T, (np.size(v)))

        print("\nVelocities:")
        print("----------------------------------------------")
        print("Max velocity:", np.max(np.sqrt(Vx * Vx + Vy * Vy)))
        print("Net flow x:", flowx)
        print("Net flow y:", flowy)
        print("Net flow:", flow)
        print("Net flow x:", flowx)
        print("Net flow y:", flowy)
        print("Net flow:", flow)

        if vLy * Lx == vLx * Ly:
            print("The velocity field is conservative")
            print("-----> vLy * Lx == vLx * Ly <-----")
        else:
            raise ValueError(
                "The velocity field is not conservative! vLy*Lx must be equal to vLx*Ly"
            )
        print(f"Vvx sum: {np.sum(Vvx):.4g} m/s")
        print(f"Vvy sum: {np.sum(Vvy):.4g} m/s")
        print("----------------------------------------------")

        np.savetxt("input_velocity_%d.txt" % (cont_v), v, header="v1\nv2\nv3\nv4")

        plt.figure()

        plt.quiver(
            X[::10, ::10] / 1000.0,
            Y[::10, ::10] / 1000.0,
            Vx[::10, ::10],
            Vy[::10, ::10],
        )

        plt.title(f"Velocity Setting: {phi}$^\circ$", fontsize=14)
        plt.ylabel("$y$ (km)", fontsize=14)
        plt.xlabel("$x$ (km)", fontsize=14)
        plt.ylim(-Ly / 1000, 0)
        plt.xlim(0, Lx / 1000)

        plt.axis("equal")

        plt.savefig(
            f"vectors_{model_name}_%d.png" % (cont_v), bbox_inches="tight", dpi=200
        )

        plt.close()

        cont_v += 1
