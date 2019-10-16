import pandas as pd
import numpy as np
import argparse
import constants


def plot_3D_params(df):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xs = df.loc[:, "param0"].values
    ys = df.loc[:, "param1"].values
    zs = df.loc[:, "param2"].values
    ax.scatter(xs, ys, zs)

    ax.set_xlabel('param0')
    ax.set_ylabel('param1')
    ax.set_zlabel('param2')

    plt.show()


def federate_characteristics(estimated_parameters, all_site_characteristics,
                             characteristics_to_use=constants.DEFAULT_CHARACTERISTICS,
                             characteristics_to_log=constants.DEFAULT_LOG_CHARACTERISTICS, output_name="merged"):

    merged_df = pd.merge(estimated_parameters, all_site_characteristics[characteristics_to_use]
                         , left_on="reach_id", right_on="NZReach")
    merged_df.drop(columns=["NZReach", "Unnamed: 0"])

    selected_characteristics = merged_df[characteristics_to_use + ["distribution", "param0", "param1", "param2"]]

    # apply log to given characteristics
    for characteristic in characteristics_to_log:
        new_name = "log10{}".format(characteristic)
        selected_characteristics.loc[:, new_name] = np.log10(selected_characteristics[characteristic])

    selected_characteristics.drop(columns=characteristics_to_log)
    selected_characteristics.to_csv("{}.csv".format(output_name))


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("estimated_fdc")
    parser.add_argument("site_characteristics")
    parser.add_argument("--output_name", type=str, default="merged")
    return parser.parse_args()


args = parse_arguments()

estimated_parameters = pd.read_csv(args.estimated_fdc)
all_site_characteristics = pd.read_csv(args.site_characteristics)
output_name = args.output_name

plot_3D_params(estimated_parameters)

federate_characteristics(estimated_parameters, all_site_characteristics, output_name=args.output_name)