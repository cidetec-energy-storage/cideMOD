#
# Copyright (c) 2023 CIDETEC Energy Storage.
#
# This file is part of cideMOD.
#
# cideMOD is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>.
#
import pandas as pd

from IPython.display import display
import plotly.graph_objects as go
import ipywidgets as widgets
from pathlib import Path

from cideMOD.helpers.baseview import BaseView, STYLE, LAYOUT


class PlotView(BaseView):
    def __init__(self, problem, results_path=None):
        super().__init__()
        self.problem = problem
        if not results_path:
            results_path = self.problem.save_path
        results_file = Path(results_path).joinpath("condensated.txt")
        assert results_file.exists(), "Results not found in the save_path"
        self.data = pd.read_csv(results_file, sep='\t')
        self.data.rename(columns={'# Time [s]': 'Time [s]'}, inplace=True)
        self.name_plot = "Results"

        # Define timescale
        t_type = 'Time [s]'
        time_divisions = {'h': 3600, 'days': 24, 'months': 30}
        for k, value in time_divisions.items():
            if self.data[t_type].max() > value * 4:
                self.data[t_type] = self.data[t_type] / value
                self.data.rename(columns={t_type: f'Time [{k}]'}, inplace=True)
                t_type = f'Time [{k}]'
            else:
                break

        self.data_fields = list(self.data)
        uniq_var_x = self.data_fields
        uniq_var_y = self.data_fields

        # OPTIONS:
        x = t_type
        self.dropdown_x = widgets.Dropdown(
            options=uniq_var_x, description="x", value=x, layout=LAYOUT, style=STYLE
        )
        self.dropdown_x.observe(self.dropdown_x_handler, "value")
        y = self.data_fields[1]
        self.dropdown_y = widgets.Dropdown(
            options=uniq_var_y, description="y", value=y, layout=LAYOUT, style=STYLE
        )
        self.dropdown_y.observe(self.dropdown_y_handler, "value")

        self.controls_options = widgets.Accordion(
            [
                widgets.HBox(
                    [
                        widgets.VBox([self.dropdown_x, self.dropdown_y]),
                    ]
                )
            ],
            selected_index=None,
        )
        self.controls_options.set_title(0, "Options")

        self.accordion = widgets.VBox([self.controls_options, self.controls_layout])
        self.vbox_display = widgets.VBox([self.accordion])
        display(self.vbox_display)

        self.fig = go.FigureWidget()
        self._plot_wrapper(
            self.dropdown_x.value,
            self.dropdown_y.value,
        )
        self.figure_standard_layout(self.fig)
        self.vbox_display.children = [*self.vbox_display.children, self.fig]

    def dropdown_x_handler(self, change):
        self.fig.layout.xaxis.title = str(change.new)
        self.text_xlabel.value = str(change.new)
        self._plot_wrapper(
            change.new,
            self.dropdown_y.value,
        )

    def dropdown_y_handler(self, change):
        self.fig.layout.yaxis.title = str(change.new)
        self.text_ylabel.value = str(change.new)
        self._plot_wrapper(
            self.dropdown_x.value,
            change.new,
        )

    def _plot_wrapper(self, x, y):
        self.fig.data = []
        self._plot(self.data[x], self.data[y])
        self.text_xlabel.value = str(x)
        self.fig.layout.xaxis.title = self.text_xlabel.value
        self.text_ylabel.value = str(y)
        self.fig.layout.yaxis.title = self.text_ylabel.value
        self.fig.layout.title = self.text_title.value

    def _plot(self, data_x, data_y):
        self.fig.add_scattergl(x=data_x, y=data_y, name=self.name_plot, showlegend=True)
        self.fig.data[-1].on_click(self.open_line_properties)
