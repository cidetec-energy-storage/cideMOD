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
import ipywidgets as widgets

pd.options.mode.chained_assignment = None  # default='warn'

COLOR_PALETTE = [
    "#636EFA",
    "#EF553B",
    "#00CC96",
    "#AB63FA",
    "#FFA15A",
    "#19D3F3",
    "#FF6692",
    "#B6E880",
    "#FF97FF",
    "#FECB52",
]
STYLE = {"description_width": "50px"}
LAYOUT = {"width": "200px"}


class BaseView:
    def __init__(self) -> None:
        # LAYOUT:
        checkbox_fig_size = widgets.Checkbox(
            value=False, description="Custom figure size", layout=LAYOUT, style=STYLE
        )
        checkbox_fig_size.observe(self.autosize_eventhandler, "value")
        self.width = widgets.Text(
            value="800",
            description="Width: ",
            continuous_update=False,
            disabled=True,
            layout=LAYOUT,
            style=STYLE,
        )
        self.width.observe(self.update_width, "value")
        self.height = widgets.Text(
            value="500",
            description="Height: ",
            continuous_update=False,
            disabled=True,
            layout=LAYOUT,
            style=STYLE,
        )
        self.height.observe(self.update_height, "value")

        self.text_xlabel = widgets.Text(
            value="x", description="xlabel", continuous_update=False, layout=LAYOUT, style=STYLE
        )
        self.text_xlabel.observe(self.update_xlabel, "value")
        self.text_ylabel = widgets.Text(
            value="y", description="ylabel", continuous_update=False, layout=LAYOUT, style=STYLE
        )
        self.text_ylabel.observe(self.update_ylabel, "value")
        self.text_title = widgets.Text(
            value="Figure",
            description="title",
            continuous_update=False,
            layout=LAYOUT,
            style=STYLE,
        )
        self.text_title.observe(self.update_title, "value")

        self.controls_layout = widgets.Accordion(
            [
                widgets.HBox(
                    [
                        widgets.VBox([checkbox_fig_size, self.width, self.height]),
                        widgets.VBox([self.text_xlabel, self.text_ylabel, self.text_title]),
                    ]
                )
            ],
            selected_index=None,
        )
        self.controls_layout.set_title(0, "Layout")

        # LINE OPTIONS
        self.plot_mode = widgets.RadioButtons(
            options=["lines", "markers", "lines+markers", "dashes"],
            description="Plot: ",
            layout=LAYOUT,
            style=STYLE,
        )
        self.plot_mode.observe(self.plot_mode_handler, "value")
        self.legend_label = widgets.Text(value="", placeholder="Type legend name", description="")
        enter = widgets.Button(description="Ok")
        enter.on_click(self.line_properties)
        color_buttons = []
        for i, color in enumerate(COLOR_PALETTE):
            color_buttons.append(
                widgets.Button(
                    style=dict(text_color=color, button_color=color),
                    layout=widgets.Layout(width="20px"),
                )
            )
            color_buttons[i].value = color
            color_buttons[i].on_click(self.update_color)

        self.controls_line = widgets.Accordion(
            [
                widgets.VBox(
                    [
                        self.plot_mode,
                        widgets.HBox([self.legend_label, widgets.HBox(children=color_buttons)]),
                        enter,
                    ]
                )
            ],
            selected_index=None,
        )
        self.controls_line.set_title(0, "Line properties")

    def figure_standard_layout(self, fig):
        fig.update_layout(plot_bgcolor="#FFFFFF")
        fig.update_xaxes(
            linecolor="black",
            mirror=True,
            gridcolor="lightgrey",
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=0.75,
        )
        fig.update_yaxes(
            linecolor="black",
            mirror=True,
            gridcolor="lightgrey",
            zeroline=True,
            zerolinecolor="black",
            zerolinewidth=0.75,
        )

    def update_xlabel(self, change):
        self.fig.layout.xaxis.title = change.new

    def update_ylabel(self, change):
        self.fig.layout.yaxis.title = change.new

    def update_title(self, change):
        self.fig.layout.title = change.new

    def update_width(self, change):
        self.fig.update_layout(
            autosize=False,
            width=int(change.new),
            height=int(self.height.value),
        )

    def update_height(self, change):
        self.fig.update_layout(
            autosize=False,
            width=int(self.width.value),
            height=int(change.new),
        )

    def autosize_eventhandler(self, change):
        self.fig.update_layout(
            autosize=not change.new,
            width=int(self.width.value),
            height=int(self.height.value),
        )
        self.width.disabled = not change.new
        self.height.disabled = not change.new

    def open_line_properties(self, lines, points, select, fig=None):
        if points.point_inds:
            if fig:
                self.fig_index = fig
            else:
                self.fig_index = self.fig
            self.data_index = points.trace_index
            label = points.trace_name
            self.plot_mode.value = (
                "dashes"
                if self.fig_index.data[self.data_index].line["dash"] == "dash"
                else self.fig_index.data[self.data_index].mode
            )
            self.legend_label.value = label
            if not any([ch._titles["0"] == "Line properties" for ch in self.accordion.children]):
                self.accordion.children = [*self.accordion.children, self.controls_line]

    def plot_mode_handler(self, change):
        with self.fig_index.batch_update():
            if change.new == "dashes":
                self.fig_index.data[self.data_index].mode = "lines"
                self.fig_index.data[self.data_index].line = dict(dash="dash")
            else:
                self.fig_index.data[self.data_index].mode = change.new
                self.fig_index.data[self.data_index].line = dict(dash=None)

    def update_color(self, change):
        with self.fig_index.batch_update():
            self.fig_index.data[self.data_index].line.color = change.value

    def line_properties(self, change):
        with self.fig_index.batch_update():
            self.fig_index.data[self.data_index].name = self.legend_label.value
        self.accordion.children = tuple(
            item for item in self.accordion.children if item._titles["0"] != "Line properties"
        )
        self.controls_line.selected_index = None
