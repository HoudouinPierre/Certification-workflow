import matplotlib.pyplot as pl
# import plotly.express as px
import pandas as pd


class PlotSimpleFakeFunction:

    def __init__(self, simple_black_box_function):
        self._simple_black_box_function = simple_black_box_function

    def prepare_plots(self):
        self._simple_black_box_function.compute_normalization_quantile()

    def matplotlib_plot_empirical_distribution(self, save_path=None):
        pl.figure(figsize=(20, 13))
        pl.clf()
        pl.hist(self._simple_black_box_function.normalized_max_empirical_distribution, bins=200, color="gold", edgecolor="red")
        pl.grid()
        pl.xlabel("Maximum"               , fontsize=20)
        pl.ylabel("Empirical distribution", fontsize=20)
        pl.axvline(x=1, linewidth=5, linestyle="dashed", color="green", label="IST")
        pl.legend()
        pl.title("Black-box empirical distribution", fontsize=24)
        if save_path:
            pl.savefig(save_path + "Black-box empirical distribution.png")
            pl.close()

    def plotly_plot_empirical_distribution(self, save_path=None):
        df = pd.DataFrame(self._simple_black_box_function.normalized_max_empirical_distribution, columns=["Samples"])
        fig = px.histogram(df, x="Samples")
        fig.update_xaxes (tickprefix="<b>", ticksuffix="<b><br>")
        fig.update_yaxes (tickprefix="<b>", ticksuffix="<b><br>")
        fig.update_layout(width=1050, height=700, title="Black-box function empirical distribution", template="plotly_white")
        fig.show()
        if save_path:
            fig.write_image(save_path + "Black-box empirical distribution.png")
