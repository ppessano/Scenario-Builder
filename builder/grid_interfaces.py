"""This module contains the functions that deal with creating the grid and its interfaces"""
import os
import ast
import numpy as np
import pandas as pd


def transform_coordinates(
    coordinates_file: str or object, structure: str = "Shear_Zone" | "Terrane"
):
    """
    Transform decimal coordinates into metric coordinates

    Parameters
    ----------------------------------------------------------------------------
    coordinates_file: str or object
        file containing the coordinates
    structure: str
        Type of structure in the coordinates_file
        Options are:
        - "Shear_Zone"
        - "Terrane"
    Return
    ----------------------------------------------------------------------------
    CSV file containing the transformed coordinates
    """
    fname = os.path.splitext(coordinates_file)[0]

    df = pd.read_csv(coordinates_file)
    df["xcoord"] = abs((((40.89 - abs(df["xcoord"])) / 0.01) * 1.09).round(2))
    df["ycoord"] = abs((((abs(df["ycoord"]) - 5.48) / 0.01) * 1.06).round(2))
    df = df.sort_values(by=["id", "xcoord"], ascending=True)

    new_df = pd.DataFrame(columns=["id", structure, "xcoord", "ycoord"])
    new_df["id"] = df["id"].unique()
    new_df[structure] = df[structure].unique()

    xcoord_list = []
    ycoord_list = []

    for struct in new_df[structure]:
        xcoord = [x for x in df[df[structure] == struct]["xcoord"]]
        ycoord = [y for y in df[df[structure] == struct]["ycoord"]]

        xcoord_list.append(xcoord)
        ycoord_list.append(ycoord)

    new_df["xcoord"] = xcoord_list
    new_df["ycoord"] = ycoord_list

    new_df.to_csv(f"{fname}" + "_transformed.csv")

    print("DataFrame successfully transformed!")


def create_horizontal_grid_2D(
    x_size: int or float,
    y_size: int or float,
    az_width: int or float,
    Nx: int,
    Ny: int,
):
    """
    Create the 2D grid in which the interfaces will be plotted.

    Parameters
    ----------------------------------------------------------------------------
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

    assert [x_size, y_size, az_width, Nx, Ny] > [
        0,
        0,
        0,
        0,
        0,
    ], "The variables that make the grid must be positive"

    Lx = x_size + 2 * az_width
    Ly = y_size + 2 * az_width

    x = np.linspace(0, Lx, Nx)
    y = np.linspace(-Ly, 0, Ny)

    X, Y = np.meshgrid(x, y)

    print("\nGrid Dimensions:")
    print("----------------------------------------------")
    print(
        f"Minimum X: {np.min(X)} Km, Maximum X: {np.max(X) / 1000} Km \
        \nMinimum Y: {np.min(Y) / 1000} Km, Maximum Y: {np.max(Y)} Km \
        \nNx: {Nx}, Ny: {Ny}"
    )
    print("----------------------------------------------")

    return Lx, Ly, x, y, X, Y


def create_interfaces_2D(
    sz_coords: str or object,
    sz_width: int or float,
    sub_width: int or float,
    az_width: int or float,
    x: np.array,
    x_size: int or float,
    Nx: int,
    Lx: int or float,
    Ly: int or float,
    tr_coords: str or object = None,
):
    """
    Create the interfaces dictionary that will receive the rheological parameters later

    Parameters
    ----------
    sz_coords: str or object
        File containing the metric coordinates of the shear zones
    sz_width: int or float
        Thickness of the shear zones (m)
        Must be a positive and non-zero value
    sub_width: int or float
        Thickness of the subsidiary shear zones (m)
        Must be a positive and non-zero value
    az_width: integer or float
        Thickness of the accommodation zone (m)
        Must be a positive and non-zero value
    x: np.array
        Horizontal (x) coordinates array
    Nx: int
        Number of nodes in the x direction
    Lx: int or float
        Length of the x direction (m)
    Ly: int or float
        Length of the y direction (m)
    tr_coords: str or object (optional, default is None)
        File containing the metric coordinates of the terranes

    Returns
    -------
    interfaces: dict
        Dictionary containing the coordinates of each interface. The interfaces are ordered from bottom
        to top.
    """
    labels = []
    layers = []

    # Converting the coordinates into arrays of numbers:
    sz_coords["xcoord"] = sz_coords["xcoord"].apply(lambda s: list(ast.literal_eval(s)))
    sz_coords["ycoord"] = sz_coords["ycoord"].apply(lambda s: list(ast.literal_eval(s)))
    for i in range(len(sz_coords)):
        sz_coords["xcoord"][i] = np.array(
            sz_coords["xcoord"][i], dtype="float32"
        ).round(2)
        sz_coords["ycoord"][i] = np.array(
            sz_coords["ycoord"][i], dtype="float32"
        ).round(2)

    for i in range(len(sz_coords)):

        y_bot = sz_coords["ycoord"][i] * -1000 - az_width
        x_bot = sz_coords["xcoord"][i] * 1000 + az_width

        if sz_coords["Shear_Zone"][i].startswith("Subsidiary"):
            y_top = np.copy(y_bot)
            y_top[1:-1] += sub_width
            x_top = np.copy(x_bot)

        else:
            y_top = np.copy(y_bot)
            y_top[1:-1] += sz_width
            x_top = np.copy(x_bot)

        bot_coord = np.interp(x, x_bot, y_bot)
        top_coord = np.interp(x, x_top, y_top)

        # Labels -> name of each interface:
        labels += [
            sz_coords["Shear_Zone"][i] + " Bottom",
            sz_coords["Shear_Zone"][i] + " Top",
        ]
        # Layers -> coordinate values of each interface:
        layers += [bot_coord, top_coord]

    # Adding the accommodation zone at both ends of the list:
    labels.insert(0, "Accommodation Zone Bottom")
    labels.append("Accommodation Zone Top")

    layers.insert(0, np.ones(Nx) * -(Ly - az_width))
    layers.append(np.ones(Nx) * -az_width)

    interfaces_order = [
        "Accommodation Zone Bottom",
        "Pernambuco Bottom",
        "Pernambuco Top",
        "Subsidiary 1 Bottom",
        "Subsidiary 1 Top",
        "Subsidiary 2 Bottom",
        "Subsidiary 2 Top",
        "Patos Bottom",
        "Patos Top",
        "Subsidiary 3 Bottom",
        "Subsidiary 3 Top",
        "Tatajuba-Jaguaribe Bottom",
        "Tatajuba-Jaguaribe Top",
        "Orós Bottom",
        "Orós Top",
        "Senador-Pompeu Bottom",
        "Senador-Pompeu Top",
        "Accommodation Zone Top",
    ]

    if tr_coords is not None:
        tr_coords["xcoord"] = tr_coords["xcoord"].apply(
            lambda s: list(ast.literal_eval(s))
        )
        tr_coords["ycoord"] = tr_coords["ycoord"].apply(
            lambda s: list(ast.literal_eval(s))
        )

        for i in range(len(tr_coords)):
            tr_coords["xcoord"][i] = np.array(
                tr_coords["xcoord"][i], dtype="float32"
            ).round(2)
            tr_coords["ycoord"][i] = np.array(
                tr_coords["ycoord"][i], dtype="float32"
            ).round(2)

        for i in range(len(tr_coords)):
            if "Bottom" in tr_coords["Terrane"][i]:
                tr_bottom = tr_coords["Terrane"][i]
                y_bot = tr_coords["ycoord"][i] * -1000 - az_width
                x_bot = tr_coords["xcoord"][i] * 1000 + az_width

                labels.append(tr_bottom)

                bot_coord = np.interp(x, x_bot, y_bot)
                layers.append(bot_coord)

        for i in range(len(tr_coords)):
            if "Top" in tr_coords["Terrane"][i]:
                tr_top = tr_coords["Terrane"][i]
                y_top = tr_coords["ycoord"][i] * -1000 - az_width
                x_top = tr_coords["xcoord"][i] * 1000 + az_width

                labels.append(tr_top)

                top_coord = np.interp(x, x_top, y_top)
                layers.append(top_coord)

        interfaces_order = [
            "Accommodation Zone Bottom",
            "Cachoeirinha-Seridó Bottom 1",
            "Cachoeirinha-Seridó Top 1",
            "Pernambuco Bottom",
            "Pernambuco Top",
            "Subsidiary 1 Bottom",
            "Subsidiary 1 Top",
            "Cariris-Velhos Bottom",
            "Subsidiary 2 Bottom",
            "Subsidiary 2 Top",
            "Cachoeirinha-Seridó Bottom 2",
            "Cachoeirinha-Seridó Top 2",
            "Patos Bottom",
            "Patos Top",
            "Cachoeirinha-Seridó Bottom 3",
            "Cachoeirinha-Seridó Top 3a",
            "Subsidiary 3 Bottom",
            "Subsidiary 3 Top",
            "Cachoeirinha-Seridó Top 3b",
            "Tatajuba-Jaguaribe Bottom",
            "Tatajuba-Jaguaribe Top",
            "Orós Bottom",
            "Orós Top",
            "Senador-Pompeu Bottom",
            "Senador-Pompeu Top",
            "Accommodation Zone Top",
        ]

    # Building the dictionary:
    dict_interfaces = dict(zip(labels, layers))

    # Reordering the dictionary:
    interfaces = {k: dict_interfaces[k] for k in interfaces_order}

    print("\nInterfaces:")
    print("----------------------------------------------")
    for label, layer in interfaces.items():
        interfaces[label][x <= Lx * (az_width / Lx)] = 0
        interfaces[label][x >= Lx * (az_width + x_size) / Lx] = 0
        print(f"{label}: {np.size(layer)} nodes")
    print("----------------------------------------------")

    return interfaces


def apply_parameters(
    interfaces: dict,
    az_params: list,
    host_params: list,
    sz_params: list,
    tr_params: list,
):
    """
    Apply rheological parameters to the interfaces.
    The input parameters correspond to the following list:
        [C, rho, H, A, n, Q, V]
        - C: scale factor
        - rho: density (kg/m³)
        - H: radiogenic heat production (W/kg)
        - A: pre-exponential factor (Pa^(-n)/s)
        - Q: activation energy (J/mol)
        - V: activation volume (m³/mol)

    Parameters
    -----------------------------------------------------------------------
    interfaces: dict
        Dictionary ordered containing the coordinates of the interfaces
    az_params: list
        File containing the metric coordinates of the shear zones
    host_params: list
        Thickness of the shear zones (m)
        Must be a positive and non-zero value
    sz_params: list
        Thickness of the subsidiary shear zones (m)
        Must be a positive and non-zero value
    tr_params: list
        Thickness of the accommodation zone (m)
        Must be a positive and non-zero value

    Returns
    -----------------------------------------------------------------------
    interfaces_file: txt file
        Text file containing the interfaces coordinates and their rheological
        parameters
    """
    with open("interfaces.txt", "w") as f:
        #               0                1               2               3               4               5               6             7                8             9               10            11               12           13                14             15               16              17             18                19            20              21               22              23              24              25              26
        layer_properties = f"""             
                C   {az_params[0]} {host_params[0]} {tr_params[0]} {tr_params[0]} {sz_params[0]} {host_params[0]} {sz_params[0]} {host_params[0]} {tr_params[0]} {sz_params[0]} {tr_params[0]} {tr_params[0]} {host_params[0]} {sz_params[0]} {host_params[0]} {tr_params[0]} {host_params[0]} {sz_params[0]} {tr_params[0]} {host_params[0]} {sz_params[0]} {host_params[0]} {sz_params[0]} {host_params[0]} {sz_params[0]} {host_params[0]} {az_params[0]}
                rho {az_params[1]} {host_params[1]} {tr_params[1]} {tr_params[1]} {sz_params[1]} {host_params[1]} {sz_params[1]} {host_params[1]} {tr_params[1]} {sz_params[1]} {tr_params[1]} {tr_params[1]} {host_params[1]} {sz_params[1]} {host_params[1]} {tr_params[1]} {host_params[1]} {sz_params[1]} {tr_params[1]} {host_params[1]} {sz_params[1]} {host_params[1]} {sz_params[1]} {host_params[1]} {sz_params[1]} {host_params[1]} {az_params[1]}
                H   {az_params[2]} {host_params[2]} {tr_params[2]} {tr_params[2]} {sz_params[2]} {host_params[2]} {sz_params[2]} {host_params[2]} {tr_params[2]} {sz_params[2]} {tr_params[2]} {tr_params[2]} {host_params[2]} {sz_params[2]} {host_params[2]} {tr_params[2]} {host_params[2]} {sz_params[2]} {tr_params[2]} {host_params[2]} {sz_params[2]} {host_params[2]} {sz_params[2]} {host_params[2]} {sz_params[2]} {host_params[2]} {az_params[2]}
                A   {az_params[3]} {host_params[3]} {tr_params[3]} {tr_params[3]} {sz_params[3]} {host_params[3]} {sz_params[3]} {host_params[3]} {tr_params[3]} {sz_params[3]} {tr_params[3]} {tr_params[3]} {host_params[3]} {sz_params[3]} {host_params[3]} {tr_params[3]} {host_params[3]} {sz_params[3]} {tr_params[3]} {host_params[3]} {sz_params[3]} {host_params[3]} {sz_params[3]} {host_params[3]} {sz_params[3]} {host_params[3]} {az_params[3]}
                n   {az_params[4]} {host_params[4]} {tr_params[4]} {tr_params[4]} {sz_params[4]} {host_params[4]} {sz_params[4]} {host_params[4]} {tr_params[4]} {sz_params[4]} {tr_params[4]} {tr_params[4]} {host_params[4]} {sz_params[4]} {host_params[4]} {tr_params[4]} {host_params[4]} {sz_params[4]} {tr_params[4]} {host_params[4]} {sz_params[4]} {host_params[4]} {sz_params[4]} {host_params[4]} {sz_params[4]} {host_params[4]} {az_params[4]}
                Q   {az_params[5]} {host_params[5]} {tr_params[5]} {tr_params[5]} {sz_params[5]} {host_params[5]} {sz_params[5]} {host_params[5]} {tr_params[5]} {sz_params[5]} {tr_params[5]} {tr_params[5]} {host_params[5]} {sz_params[5]} {host_params[5]} {tr_params[5]} {host_params[5]} {sz_params[5]} {tr_params[5]} {host_params[5]} {sz_params[5]} {host_params[5]} {sz_params[5]} {host_params[5]} {sz_params[5]} {host_params[5]} {az_params[5]}
                V   {az_params[6]} {host_params[6]} {tr_params[6]} {tr_params[6]} {sz_params[6]} {host_params[6]} {sz_params[6]} {host_params[6]} {tr_params[6]} {sz_params[6]} {tr_params[6]} {tr_params[6]} {host_params[6]} {sz_params[6]} {host_params[6]} {tr_params[6]} {host_params[6]} {sz_params[6]} {tr_params[6]} {host_params[6]} {sz_params[6]} {host_params[6]} {sz_params[6]} {host_params[6]} {sz_params[6]} {host_params[6]} {az_params[6]} 
            """

    for line in layer_properties.split("\n"):
        line = line.strip()
        if len(line):
            f.write(" ".join(line.split()) + "\n")

    data = np.array(tuple(interfaces.values())).T
    np.savetxt(f, data, fmt="%.1f")
