"""This module contains the functions to generate the plots"""
import os
import imageio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_map_layout(
    model_name: str,
    x: np.array,
    Lx: int or float,
    Ly: int or float,
    az_width: int or float,
    interfaces: dict,
):
    """
    Plot the layout of the model based on the interfaces file and its coordinates

    Parameters:
    -----------
    model_name: str
        Name of the model
    x: np.array
        1D coordinate array along the x-axis
    Lx: int or float
        Lenght of the x direction (m)
    Ly: int or float
        Lenght of the y direction (m)
    az_width: float
        Width of the accommodation zone in meters
    interfaces: dict
        Dictionary containing the coordinates of each interface
    """
    plt.figure(figsize=(12, 8))

    # Model total area outline:
    plt.plot(
        [0, Lx / 1000, Lx / 1000, 0, 0], [0, 0, -Ly / 1000, -Ly / 1000, 0], "k", lw=4
    )

    # Accommodation zone area:
    plt.fill(
        [0, Lx / 1000, Lx / 1000, 0, 0],
        [0, 0, -Ly / 1000, -Ly / 1000, 0],
        "aliceblue",
        [
            az_width / 1000,
            (Lx - az_width) / 1000,
            (Lx - az_width) / 1000,
            az_width / 1000,
            az_width / 1000,
        ],
        [
            -az_width / 1000,
            -az_width / 1000,
            -(Ly - az_width) / 1000,
            -(Ly - az_width) / 1000,
            -az_width / 1000,
        ],
        "white",
    )

    for label, layer in interfaces.items():
        if label.startswith("Accommodation"):
            plt.plot(x / 1000, layer / 1000, "black", lw=4.0)
        elif label.startswith("Subsidiary") and label.endswith("Bottom"):
            plt.plot(x / 1000, layer / 1000, "k", alpha=0.6, lw=1.5)
        elif (
            not label.startswith("Subsidiary")
            and not label.startswith("Cariris")
            and label.endswith("Bottom")
        ):
            plt.plot(x / 1000, layer / 1000, lw=4.0, label=f"{label}"[:-6], alpha=0.7)

    cs_color = "darkseagreen"
    cv_color = "khaki"

    # Cachoeirinha-Seridó
    plt.fill_between(
        x / 1000,
        interfaces["Cachoeirinha-Seridó Bottom 1"] / 1000,
        interfaces["Cachoeirinha-Seridó Top 1"] / 1000,
        color=cs_color,
        label="Neoproterozoic supracrustals",
    )
    plt.fill_between(
        x / 1000,
        interfaces["Cachoeirinha-Seridó Bottom 2"] / 1000,
        interfaces["Cachoeirinha-Seridó Top 2"] / 1000,
        color=cs_color,
    )
    plt.fill_between(
        x / 1000,
        interfaces["Cachoeirinha-Seridó Bottom 3"] / 1000,
        interfaces["Cachoeirinha-Seridó Top 3a"] / 1000,
        color=cs_color,
    )
    plt.fill_between(
        x / 1000,
        interfaces["Subsidiary 3 Bottom"] / 1000,
        interfaces["Cachoeirinha-Seridó Top 3b"] / 1000,
        color=cs_color,
    )

    # Cariris-Velhos:
    plt.fill_between(
        x / 1000,
        interfaces["Cachoeirinha-Seridó Top 1"] / 1000,
        interfaces["Pernambuco Bottom"] / 1000,
        color=cv_color,
        label="Tonian supracrustals",
    )
    plt.fill_between(
        x / 1000,
        interfaces["Subsidiary 2 Bottom"] / 1000,
        interfaces["Cachoeirinha-Seridó Bottom 2"] / 1000,
        color=cv_color,
    )
    plt.fill_between(
        x / 1000,
        interfaces["Cariris-Velhos Bottom"] / 1000,
        interfaces["Cachoeirinha-Seridó Bottom 2"] / 1000,
        color=cv_color,
    )

    # Annotations:
    plt.annotate(
        "N",
        xy=(680, -120),
        xytext=(680, -180),
        arrowprops=dict(facecolor="black", width=5, headwidth=15),
        ha="center",
        va="center",
        fontsize=20,
        xycoords="data",
    )
    plt.annotate(
        "Accommodation Zone",
        xy=(50, -300),
        xytext=(50, -400),
        xycoords="data",
        rotation="vertical",
        fontsize=13,
        fontstyle="italic",
    )
    plt.annotate(
        "Cachoeirinha",
        xy=(290, -360),
        xytext=(290, -360),
        xycoords="data",
        rotation=28,
        fontsize=11.5,
        fontstyle="normal",
    )
    plt.annotate(
        "Seridó",
        xy=(570, -220),
        xytext=(570, -220),
        xycoords="data",
        rotation=60,
        fontsize=11.5,
        fontstyle="normal",
    )
    plt.annotate(
        "Cariris-Velhos",
        xy=(350, -380),
        xytext=(350, -380),
        xycoords="data",
        rotation=28,
        fontsize=11.5,
        fontstyle="normal",
    )

    plt.title(f"Structural Layout\n{model_name}", fontsize=14)
    plt.ylabel("$y$ (km)", fontsize=14)
    plt.xlabel("$x$ (km)", fontsize=14)
    plt.ylim(-Ly / 1000, 0)
    plt.xlim(0, Lx / 1000)

    handles, labels = plt.gca().get_legend_handles_labels()
    legend = plt.legend(
        reversed(handles),
        reversed(labels),
        fontsize=10,
        ncol=4,
        framealpha=1,
        loc="lower right",
    )
    legend.get_title().set_fontsize("11")
    legend.get_frame().set_edgecolor("k")

    plt.savefig(f"interfaces_{model_name}", bbox_inches="tight", dpi=200)
    plt.show()


def shadow_plot(
    first_step: int,
    final_step: int,
    step: int = 10,
    az_width: float = 100e3,
    remove_az: bool = True,
    savefig: bool = True,
    create_animation: bool = True,
):
    """
    Generate a plot that takes the density as a background and
    the strain rate as a shadow to show the deformation.

    Parameters:
    -----------
    first_step: int
        First step to be plotted.
    final_step: int
        Final step to be plotted.
    step: int
        Step between each plot.
    az_width: float
        Width of the accommodation zone in meters (default is 100 km).
    remove_az: bool
        Remove the accommodation zone from the plot (default is True).
    savefig: bool
        If True, save the figure.
    create_animation: bool
        If True, create an animation.
    """
    with open("param.txt", "r") as f:
        line = f.readline()
        line = line.split()
        Nx = int(line[2])
        line = f.readline()
        line = line.split()
        Ny = int(line[2])
        line = f.readline()
        line = line.split()
        Lx = float(line[2])
        line = f.readline()
        line = line.split()
        Ly = float(line[2])

    if remove_az:
        xlim = Lx - az_width
        ylim = Ly - az_width
    else:
        xlim = Lx
        ylim = Ly

    # Creating coordinates and meshgrid:
    x = np.linspace(0, Lx / 1e3, Nx)
    y = np.linspace(-Ly / 1e3, 0, Ny)

    X, Y = np.meshgrid(x, y)

    for counter in range(first_step, final_step + step, step):

        time = np.loadtxt("time_" + str(counter) + ".txt", dtype="str")
        time = time[:, 2:]
        time = time.astype("float")

        time_Myr = counter * (5000 / 1e6)  # 5000, period of each interval
        time_Myr = round(time_Myr, 3)
        age_Ma = 140 - time_Myr  # Age in Ma
        age_Ma = round(age_Ma, 3)

        rho = pd.read_csv(
            "density_" + str(counter) + ".txt",
            delimiter=" ",
            comment="P",
            skiprows=2,
            header=None,
        )
        rho = rho.to_numpy()
        rho[np.abs(rho) < 1.0e-200] = 0
        rho = np.reshape(rho, (Nx, Ny), order="F")
        rho = np.transpose(rho)

        # Read strain
        strain = pd.read_csv(
            "strain_" + str(counter) + ".txt",
            delimiter=" ",
            comment="P",
            skiprows=2,
            header=None,
        )
        strain = strain.to_numpy()
        strain[np.abs(strain) < 1.0e-200] = 0
        strain = np.reshape(strain, (Nx, Ny), order="F")
        strain = np.transpose(strain)
        strain[rho < 200] = 0
        strain_log = np.log10(strain)

        print("Step =", counter)
        print(f"Elapsed Time = {time_Myr} Myr\n\n")
        # print("strain(log)", np.min(strain_log), np.max(strain_log))
        # print("strain", np.min(strain), np.max(strain))

        plt.figure(figsize=(8, 6))
        plt.title(f"Elapsed Time: {time_Myr} Myr\n Age: {age_Ma} Ma", fontsize=14)
        plt.ylabel("$y$ (km)", fontsize=12)
        plt.xlabel("$x$ (km)", fontsize=12)

        # Create the colors to plot the density
        az = "aliceblue"
        tr = "yellowgreen"
        hr = "peachpuff"
        sz = "darkred"
        colors = [az, tr, hr, sz]

        # Plot density
        plt.contourf(
            X,
            Y,
            rho,
            levels=[2500.0, 2700.0, 2750.0, 2900.0],
            extent=[0, Lx / 1e3, -Ly / 1e3, 0],
            colors=colors,
        )

        # Plot strain_log
        plt.imshow(
            strain_log[::-1, :],
            extent=[0, Lx / 1e3, -Ly / 1e3, 0],
            zorder=100,
            alpha=0.5,
            cmap=plt.get_cmap("Greys"),
            vmin=-0.5,
            vmax=0.9,
        )

        plt.xlim(az_width / 1e3, xlim / 1e3)
        plt.ylim(-ylim / 1e3, -az_width / 1e3)

        old_x_ticks = list(
            np.arange(az_width / 1e3, Lx / 1e3, az_width / 1e3).astype("int")
        )
        new_x_ticks = list(np.arange(0, xlim / 1e3, az_width / 1e3).astype("int"))
        old_y_ticks = list(np.arange(-ylim / 1e3, 0, az_width / 1e3).astype("int"))
        new_y_ticks = list(
            np.arange(
                -ylim / 1e3 + az_width / 1e3, 0 + az_width / 1e3, az_width / 1e3
            ).astype("int")
        )

        plt.xticks(old_x_ticks, new_x_ticks)
        plt.yticks(old_y_ticks, new_y_ticks)

        if savefig:
            os.makedirs(os.path.dirname("strain_shadow_plots/"), exist_ok=True)
            plt.savefig(
                "strain_shadow_plots/strain_shadow_plot_{:05}.png".format(counter * 1),
                bbox_inches="tight",
                dpi=100,
                transparent=False,
                facecolor="white",
            )

    if create_animation:
        print("Creating animation...")
        list_of_im_paths = []

        for counter in range(first_step, final_step + 10, 10):
            im_path = "strain_shadow_plots/strain_shadow_plot_{:05}.png".format(
                counter * 1
            )
            list_of_im_paths.append(im_path)

        ims = [imageio.imread(f) for f in list_of_im_paths]
        imageio.mimwrite(
            "strain_shadow_plots/strain_shadow_animation.gif", ims, duration=0.003
        )
