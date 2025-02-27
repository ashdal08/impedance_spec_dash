import os
import re
import subprocess
import threading
import time
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio
import scipy

from gevent.pywsgi import WSGIServer
import waitress

from custom_libs.hp4192a import HP4192A
# from custom_libs.dummy_gpib import HP4192A

from dash import (
    Dash,
    html,
    Input,
    Output,
    callback,
    dcc,
    clientside_callback,
    Patch,
    State,
    ClientsideFunction,
)
import dash_bootstrap_components as dbc

from dash_bootstrap_templates import load_figure_template

load_figure_template(["bootstrap", "bootstrap_dark"])

BOOTSTRAP = pio.templates["bootstrap"]
BOOTSTRAP_DARK = pio.templates["bootstrap_dark"]


def modifyColorwayInPlotlyTemplate(template: go.layout.Template) -> go.layout.Template:
    """Method to modify the colorway of the plotly Figure. Needed to get the same color for the traces in the real, imaginary and cole-cole plot.

    Parameters
    ----------
    template : plotly.graph_objects.layout.Template
        The plotly layout template to be modified.

    Returns
    -------
    plotly.graph_objects.layout.Template
        The modified plotly layout template
    """
    colorway = template["layout"]["colorway"]
    mod_colorway = []
    for color in colorway:
        for i in range(0, 3):
            mod_colorway.append(color)

    template["layout"]["colorway"] = mod_colorway
    return template


BOOTSTRAP = modifyColorwayInPlotlyTemplate(BOOTSTRAP)
"""The Bootstrap template for the plotly figure."""
BOOTSTRAP_DARK = modifyColorwayInPlotlyTemplate(BOOTSTRAP_DARK)
"""The Dark Bootstrap template for the plotly figure."""


class DataBackend:
    """Class for the GUI backend.

    Methods
    -------
    __init__:
        Method called when the class is initialized.
    multimeterGetTemperature:
        Method for the multimeter thread to get temperature from the VC 870 multimeter in Celsius.
    resetFig:
        Method to reset the plotly figure object.
    addFigTraces:
        Method to add new traces to a Plotly Figure object.
    updatePlotData:
        Method to update trace data in a plotly Figure.
    startMeasurementThread:
        Method to start the required initialize and start the required measurement threads.
    runImpedanceMeasurement:
        Method for the impedance mode measurement.
    runPermittivityMeasurement:
        Method for the permittivity mode measurement.
    prepareForShutdown:
        Method called before shutting down the application.
    """

    hp_4192a: HP4192A
    """The HP 4192A Impedance Analyzer resource on GPIB."""

    meas_running: bool = False
    """Boolean indicating if a measurement is running."""

    meas_cancelled: bool = False
    """Boolean indicating if a measurement was cancelled."""

    meas_completed: bool = True
    """Boolean indicating if a measurement has finished."""

    dataframe_data: pd.DataFrame
    """Pandas dataframe for the values during measurement."""

    plot_data_frame: pd.DataFrame
    """Pandas dataframe with values for the plot update."""

    meas_thread: threading.Thread
    """Thread object containing the required measurement function."""

    temperature_thread: threading.Thread
    """Thread object containing the required temperature function."""

    progress_current: int = 0
    """Integer indicating the current progress of the measurement. Required to update the progress bar on the web app."""

    plot_fig: go.Figure
    """Plotly figure object containing the plot of the measurement."""

    voltcraft_vc870_temperature: float
    """Temperature in Celsius recorded from the Voltcraft VC870 multimeter."""

    def __init__(self) -> None:
        """Method called when the class is initialized."""

        # ************HP 4192A***************
        # VISA setup
        # self.hp_4192a = hp4192a.HP4192A()
        self.plot_fig = self.resetFig(None)
        self.hp_4192a = HP4192A("GPIB0::1::INSTR")
        self.voltcraft_vc870_temperature = np.nan
        if self.hp_4192a.isConnected():
            self.hp_4192a.sendCommandString(
                data_ready=1,
                zy_range=-1,
                high_speed=1,
                average=1,
                output_format=0,
                function_A=0,
                function_B=1,
                circuit_mode=0,
                trigger=0,
                osc_level=1.000,
                dc_bias=False,
                execute=True,
            )
            test = self.hp_4192a.readFromInstrument()
            if type(test[0]) is not float:
                print("Error in connected instrument.")

        pass

    def multimeterGetTemperature(self):
        """Method for the multimeter thread to get temperature from the VC 870 multimeter in Celsius."""
        # Get the directory of the Python script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        sigrok_path = os.path.join(script_dir, "sigrok-cli", "sigrok-cli")  # Full path to the sigrock-cli executable
        sigrok_cmd = (
            f'"{sigrok_path}" -d voltcraft-vc870-ser:conn="hid/ch9325/raw=\\\\?\\hid#vid_1a86&pid_e008#6&36babeda&0&0000#'
            f'{{4d1e55b2-f16f-11cf-88cb-001111000030}}" --samples 1'
        )  # sigrok-cli command to be used. "sigrok-cli -d voltcraft-vc870-ser:conn="hid/ch9325/raw=\\?\hid#vid_1a86&pid_e008#6&36babeda&0&0000#{4d1e55b2-f16f-11cf-88cb-001111000030}" --samples 1"
        while self.meas_running:
            # Configure subprocess to prevent command window popup
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # Hides the command window

            # Run the subprocess silently
            process = subprocess.run(
                sigrok_cmd,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                startupinfo=startupinfo,
            )

            raw_output = process.stdout.strip()  # Get the raw output

            # Extract only the numeric value using regex
            match = re.search(r"P1:\s*([\d.]+)", raw_output)  # Matches numbers with decimal points
            if match:
                self.voltcraft_vc870_temperature = float(match.group(1))  # Extracted temperature value

    def resetFig(self, fig_2: go.Figure | None, logarithmic: bool = False, renew: bool = False) -> go.Figure:
        """Method to reset the plotly figure object.

        Parameters
        ----------
        fig_2 : plotly.graph_objects.Figure
            The plotly figure object to be modified. None when a new figure object is required.
        logarithmic : bool, optional
            Boolean indicating if the x-axis needs to be logarithmic, by default False
        renew : bool, optional
            Boolean indicating if a new plotly Figure object is needed., by default False

        Returns
        -------
        plotly.graph_objects.Figure
            The plotly figure object created.
        """
        LAYOUT = {
            "autosize": True,
            "margin": {
                "l": 80,
                "r": 5,
                "b": 40,
                "t": 40,
            },
            "dragmode": "pan",
            "font_size": 11,
            "hovermode": "closest",
            "template": BOOTSTRAP,
        }
        if renew:
            fig = make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
                subplot_titles=("Real", "Imaginary", "Cole-Cole"),
                vertical_spacing=0.1,
                figure=fig_2,
            )
        else:
            fig = make_subplots(
                rows=2,
                cols=2,
                specs=[[{}, {}], [{"colspan": 2}, None]],
                subplot_titles=("Real", "Imaginary", "Cole-Cole"),
                vertical_spacing=0.1,
            )
            """Plotly figure object for the graph area."""

            fig = self.addFigTraces(fig)

            fig.update_xaxes(
                title_text=r"$\text{Frequency [ Hz ]}$",
                row=1,
                col=1,
                type="log" if logarithmic else "linear",
            )
            fig.update_xaxes(
                title_text=r"$\text{Frequency [ Hz ]}$",
                row=1,
                col=2,
                type="log" if logarithmic else "linear",
            )
            fig.update_xaxes(
                title_text=r"$\text{Real Impedance } [ \Omega ]$",
                # type="log",
                row=2,
            )
            # Update yaxis properties
            fig.update_yaxes(
                title_text=r"$\text{Real Impedance } [ \Omega ]$",
                row=1,
                col=1,
            )
            fig.update_yaxes(
                title_text=r"$\text{Imaginary Impedance } [ \Omega ]$",
                row=1,
                col=2,
            )
            fig.update_yaxes(
                title_text=r"$\text{Imaginary Impedance } [ \Omega ]$",
                showgrid=False,
                row=2,
                col=1,
            )
            fig.update_layout(
                go.Layout(
                    **LAYOUT,
                    xaxis3={"domain": [0.25, 0.75]},
                    yaxis3={"anchor": "x3"},
                    uirevision="no_change",
                )
            )

        return fig

    def addFigTraces(self, fig: go.Figure, pass_no: int = 1) -> go.Figure:
        """Method to add new traces to a Plotly Figure object.


        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            The plotly figure object to which traces are to be added.
        pass_no : int, optional
            The pass number to be appended to the trace names, by default 1

        Returns
        -------
        plotly.graph_objects.Figure
            The plotly figure object with the new traces.
        """
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name=f"pass_{pass_no}",  # real_{pass_no}
                legendgroup=f"pass_{pass_no}",
                mode="markers+lines",
                line_width=1,
                marker_size=4,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name=f"imag_{pass_no}",
                legendgroup=f"pass_{pass_no}",
                mode="markers+lines",
                line_width=1,
                marker_size=4,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name=f"cole_{pass_no}",
                legendgroup=f"pass_{pass_no}",
                mode="markers+lines",
                line_width=1,
                marker_size=4,
                showlegend=False,
            ),
            row=2,
            col=1,
        )

        return fig

    def updatePlotData(
        self,
        fig: go.Figure,
        x_data: pd.Series,
        y1_data: pd.Series,
        y2_data: pd.Series,
        pass_no: int,
    ) -> go.Figure:
        """Method to update trace data in a plotly Figure.

        Parameters
        ----------
        fig : plotly.graph_objects.Figure
            The plotly figure object on which the trace data is to be updated.
        x_data : pandas.Series
            The series for the x axis data for the plots in the top row. In this case the Frequency in Hz.
        y1_data : pandas.Series
            The series for the y axis data of the plot on the top-left. Usually the real part of impedance or permittivity. This will be the x axis of the plot in the bottom row.
        y2_data : pandas.Series
            The series for the y axis data of the plot on the top-right. Usually the imaginary part of the impedance or permittivity. This will be the y axis data in the bottom plot.
        pass_no : int
            Integer indicating the trace name that is to be selected to be updated.

        Returns
        -------
        plotly.graph_objects.Figure
            The plotly figure with the updated trace data.

        """
        fig.update_traces(
            overwrite=False,
            x=x_data,
            y=y1_data,
            selector=dict(name=f"pass_{int(pass_no)}"),  # real_{pass_no}
        )
        fig.update_traces(
            overwrite=False,
            x=x_data,
            y=y2_data,
            selector=dict(name=f"imag_{int(pass_no)}"),
        )
        fig.update_traces(
            overwrite=False,
            x=y1_data,
            y=y2_data,
            selector=dict(name=f"cole_{int(pass_no)}"),
        )

        return fig

    def startMeasurementThread(
        self,
        start_freq: float,
        end_freq: float,
        freq_step: float,
        logarithmic: bool,
        steps_per_decade: int,
        sample_diameter: float,
        sample_thickness: float,
        stray_capacitance: float,
        get_temperature: bool,
        no_of_passes: int,
        impedance: bool = True,
    ) -> None:
        """Method to start the required initialize and start the required measurement threads.

        Parameters
        ----------
        start_freq : float
            The starting frequency for the measurement.
        end_freq : float
            The ending frequency for the measurement.
        freq_step : float
            The frequency steps in case of a linear sweep.
        logarithmic : bool
            Boolean indicating if the sweep needs to be logarithmic.
        steps_per_decade : int
            Integer indicating the number of steps per decade in case of a logarithmic sweep.
        sample_diameter : float
            The diameter of the sample (PVAC) in mm.
        sample_thickness : float
            The thickness of the sample (PVAC) in µm.
        stray_capacitance : float
            The stray capacitance value to be subtracted in farads.
        get_temperature : bool
            Boolean indicating if the sample temperature is to be recorded.
        no_of_passes : int
            Integer indicating the number of passes to be conducted during the measurement.
        impedance : bool
            Boolean indicating if the measruement is a Impedance mode measurement. If False the permittivity mode measurement is started.

        """
        if impedance:
            self.meas_thread = threading.Thread(
                target=self.runImpedanceMeasurement,
                args=[
                    start_freq,
                    end_freq,
                    freq_step,
                    logarithmic,
                    steps_per_decade,
                    get_temperature,
                    no_of_passes,
                ],
            )
        else:
            self.meas_thread = threading.Thread(
                target=self.runPermittivityMeasurement,
                args=[
                    start_freq,
                    end_freq,
                    freq_step,
                    logarithmic,
                    steps_per_decade,
                    sample_diameter,
                    sample_thickness,
                    stray_capacitance,
                    get_temperature,
                    no_of_passes,
                ],
            )

        if get_temperature:
            self.temperature_thread = threading.Thread(target=self.multimeterGetTemperature)

        self.plot_fig = self.resetFig(self.plot_fig, logarithmic)
        self.meas_thread.start()
        self.meas_running = True
        self.meas_completed = False
        if get_temperature:
            self.temperature_thread.start()

    def runImpedanceMeasurement(
        self,
        start_freq: float,
        end_freq: float,
        freq_step: float,
        logarithmic: bool,
        steps_per_decade: int,
        get_temperature: bool,
        no_of_passes: int,
    ) -> None:
        """Method for the impedance mode measurement.

        Parameters
        ----------
        start_freq : float
            The starting frequency for the measurement.
        end_freq : float
            The ending frequency for the measurement.
        freq_step : float
            The frequency steps in case of a linear sweep.
        logarithmic : bool
            Boolean indicating if the sweep needs to be logarithmic.
        steps_per_decade : int
            Integer indicating the number of steps per decade in case of a logarithmic sweep.
        get_temperature : bool
            Boolean indicating if the sample temperature is to be recorded.
        no_of_passes : int
            Integer indicating the number of passes to be conducted during the measurement.

        """
        if logarithmic:
            tot_points = (np.log10(end_freq) - np.log10(start_freq)) * steps_per_decade
            freqs_array = np.logspace(np.log10(start_freq), np.log10(end_freq), int(tot_points))
        else:
            tot_points = ((end_freq - start_freq) / freq_step) + 1
            freqs_array = [float(i) for i in np.arange(start_freq, end_freq, freq_step)]
            freqs_array.append(end_freq)

        self.progress_current = 0

        imp_z: float
        rad_theta: float
        time_stamp: str

        self.dataframe_data = pd.DataFrame({
            "Time [s]": [],
            "Pass_No.": [],
            "Temperature [°]": [],
            "Frequency [Hz]": [],
            "Real Impedance [Ω]": [],
            "Imaginary Impedance [Ω]": [],
        })

        self.plot_data_frame = pd.DataFrame({
            "Pass_No.": [],
            "Frequency [Hz]": [],
            "Real": [],
            "Imaginary": [],
        })

        i = 1  # iterator for passes
        j = 0  # iterator for array index
        progress_iterator = 0
        self.hp_4192a.sendCommandString(
            data_ready=1,
            zy_range=-1,
            high_speed=1,
            average=1,
            output_format=0,
            function_A=None,
            function_B=1,
            circuit_mode=0,
            trigger=0,
            osc_level=1.000,
            test_signal_freq=freqs_array[0],
            dc_bias=False,
            bias_setting=0.00,
            execute=True,
        )

        while i <= no_of_passes:
            j = 0
            while j < len(freqs_array):
                if self.meas_cancelled:
                    break
                self.hp_4192a.sendCommandString(
                    test_signal_freq=freqs_array[j],
                    execute=True,
                    sleep_adder=100,
                )
                time_stamp = time.strftime("%H:%M:%S")
                imp_z, rad_theta = self.hp_4192a.readFromInstrument()
                temperature = np.nan
                if get_temperature:
                    temperature = self.voltcraft_vc870_temperature
                new_data = pd.DataFrame({
                    "Time [s]": [time_stamp],
                    "Pass_No.": [i],
                    "Temperature [°]": [temperature],
                    "Frequency [Hz]": [freqs_array[j]],
                    "Real Impedance [Ω]": [imp_z * np.cos(rad_theta)],
                    "Imaginary Impedance [Ω]": [imp_z * np.sin(rad_theta)],
                })

                new_plot_point = {
                    "Pass_No.": [i],
                    "Frequency [Hz]": [freqs_array[j]],
                    "Real": [imp_z * np.cos(rad_theta)],
                    "Imaginary": [imp_z * np.sin(rad_theta)],
                }

                self.plot_data_frame = pd.concat([self.plot_data_frame, pd.DataFrame(new_plot_point)])

                self.dataframe_data = pd.concat([self.dataframe_data, new_data])

                self.data_index = len(self.dataframe_data["Time [s]"])

                progress_iterator += 1

                self.progress_current = int(progress_iterator * 100 / (tot_points * no_of_passes))

                x_data = self.plot_data_frame["Frequency [Hz]"]
                y1_data = self.plot_data_frame["Real"]
                y2_data = self.plot_data_frame["Imaginary"]

                self.plot_fig = self.updatePlotData(self.plot_fig, x_data, y1_data, y2_data, i)

                j += 1

            if self.meas_cancelled:
                self.meas_running = False
                self.meas_cancelled = False
                break
            i += 1
            self.plot_data_frame = pd.DataFrame({
                "Pass_No.": [],
                "Frequency [Hz]": [],
                "Real": [],
                "Imaginary": [],
            })
            if i <= no_of_passes:
                self.plot_fig = self.addFigTraces(self.plot_fig, i)

        time.sleep(1)
        self.meas_running = False
        self.meas_completed = True
        self.progress_current = 100
        self.hp_4192a.sendCommandString(data_ready=0, execute=True)

    def runPermittivityMeasurement(
        self,
        start_freq: float,
        end_freq: float,
        freq_step: float,
        logarithmic: bool,
        steps_per_decade: int,
        sample_diameter: float,
        sample_thickness: float,
        stray_capacitance: float,
        get_temperature: bool,
        no_of_passes: int,
    ) -> None:
        """Method for the permittivity mode measurement.

        Parameters
        ----------
        start_freq : float
            The starting frequency for the measurement.
        end_freq : float
            The ending frequency for the measurement.
        freq_step : float
            The frequency steps in case of a linear sweep.
        logarithmic : bool
            Boolean indicating if the sweep needs to be logarithmic.
        steps_per_decade : int
            Integer indicating the number of steps per decade in case of a logarithmic sweep.
        sample_diameter : float
            The diameter of the sample (PVAC) in mm.
        sample_thickness : float
            The thickness of the sample (PVAC) in µm.
        stray_capacitance : float
            The stray capacitance value to be subtracted in farads.
        get_temperature : bool
            Boolean indicating if the sample temperature is to be recorded.
        no_of_passes : int
            Integer indicating the number of passes to be conducted during the measurement.

        """
        if logarithmic:
            tot_points = (np.log10(end_freq) - np.log10(start_freq)) * steps_per_decade
            freqs_array = np.logspace(np.log10(start_freq), np.log10(end_freq), int(tot_points))
        else:
            tot_points = ((end_freq - start_freq) / freq_step) + 1
            freqs_array = [float(i) for i in np.arange(start_freq, end_freq, freq_step)]
            freqs_array.append(end_freq)

        self.progress_current = 0

        farad: float
        conductivity: float
        capacitance_0 = (
            scipy.constants.epsilon_0 * (np.pi * ((sample_diameter * 1e-3) ** 2) / 4) / (sample_thickness * 1e-6)
        )
        time_stamp: str

        self.dataframe_data = pd.DataFrame({
            "Time [s]": [],
            "Pass_No.": [],
            "Temperature [°]": [],
            "Frequency [Hz]": [],
            "ε'": [],
            'ε"': [],
        })

        self.plot_data_frame = pd.DataFrame({
            "Pass_No.": [],
            "Frequency [Hz]": [],
            "Real": [],
            "Imaginary": [],
        })

        i = 1  # iterator for passes
        j = 0  # iterator for array index
        progress_iterator = 0
        self.hp_4192a.sendCommandString(
            data_ready=1,
            zy_range=-1,
            high_speed=1,
            average=1,
            output_format=0,
            function_A=3,
            function_B=2,
            circuit_mode=2,
            trigger=0,
            osc_level=1.000,
            test_signal_freq=freqs_array[0],
            dc_bias=False,
            bias_setting=0.00,
            execute=True,
        )

        while i <= no_of_passes:
            j = 0
            while j < len(freqs_array):
                if self.meas_cancelled:
                    break
                self.hp_4192a.sendCommandString(
                    test_signal_freq=freqs_array[j],
                    execute=True,
                    sleep_adder=100,
                )
                farad, conductivity = self.hp_4192a.readFromInstrument()
                time_stamp = time.strftime("%H:%M:%S")
                ang_freq = 2 * np.pi * freqs_array[j]
                temperature = np.nan
                if get_temperature:
                    temperature = self.voltcraft_vc870_temperature
                new_data = pd.DataFrame({
                    "Time [s]": [time_stamp],
                    "Pass_No.": [i],
                    "Temperature [°]": [temperature],
                    "Frequency [Hz]": [freqs_array[j]],
                    "ε'": [(farad - stray_capacitance) / capacitance_0],
                    'ε"': [conductivity / (ang_freq * capacitance_0)],
                })

                self.new_plot_point = {
                    "Pass_No": i,
                    "Frequency [Hz]": freqs_array[j],
                    "Real": (farad - stray_capacitance) / capacitance_0,
                    "Imaginary": conductivity / (ang_freq * capacitance_0),
                }

                new_plot_point = {
                    "Pass_No.": [i],
                    "Frequency [Hz]": [freqs_array[j]],
                    "Real": [(farad - stray_capacitance) / capacitance_0],
                    "Imaginary": [conductivity / (ang_freq * capacitance_0)],
                }

                self.plot_data_frame = pd.concat([self.plot_data_frame, pd.DataFrame(new_plot_point)])

                self.dataframe_data = pd.concat([self.dataframe_data, new_data])

                self.data_index = len(self.dataframe_data["Time [s]"])

                progress_iterator += 1

                self.progress_current = int(progress_iterator * 100 / (tot_points * no_of_passes))

                x_data = self.plot_data_frame["Frequency [Hz]"]
                y1_data = self.plot_data_frame["Real"]
                y2_data = self.plot_data_frame["Imaginary"]

                self.plot_fig = self.updatePlotData(self.plot_fig, x_data, y1_data, y2_data, i)

                j += 1

            if self.meas_cancelled:
                self.meas_running = False
                self.meas_cancelled = False
                break
            i += 1
            self.plot_data_frame = pd.DataFrame({
                "Pass_No.": [],
                "Frequency [Hz]": [],
                "Real": [],
                "Imaginary": [],
            })
            if i <= no_of_passes:
                self.plot_fig = self.addFigTraces(self.plot_fig, i)

        time.sleep(1)
        self.meas_running = False
        self.meas_completed = True
        self.progress_current = 100
        self.hp_4192a.sendCommandString(data_ready=0, execute=True)

    def prepareForShutdown(self) -> None:
        """Method called before shutting down the application."""

        self.hp_4192a.sendCommandString(data_ready=0, execute=True)
        self.hp_4192a.closeConnection()
        os._exit(0)


LAYOUT: dict = {
    "autosize": True,
    "margin": {
        "l": 80,
        "r": 5,
        "b": 40,
        "t": 40,
    },
    "dragmode": "pan",
    "font_size": 11,
    "hovermode": "closest",
}
"""Dictionary describing the layout for the plot area of the dash app."""

CONFIG: dict = {
    "scrollZoom": True,
    "displaylogo": False,
    "toImageButtonOptions": {
        "format": "svg",
        "filename": "custom_image",
        "scale": 1,
    },
    "modeBarButtonsToRemove": ["select2d", "lasso2d"],
}
"""Dictionary describing the configuration of the plotly Figure object."""

THEMES: list = [
    "bootstrap",
    "cerulean",
    "cosmo",
    "cyborg",
    "darkly",
    "flatly",
    "journal",
    "litera",
    "lumen",
    "lux",
    "materia",
    "minty",
    "morph",
    "pulse",
    "quartz",
    "sandstone",
    "simplex",
    "sketchy",
    "slate",
    "solar",
    "spacelab",
    "superhero",
    "united",
    "vapor",
    "yeti",
    "zephyr",
]
"""List of theme names for the dash app."""

data_process_backend = DataBackend()
"""Backend class for the measurement processing."""


fig = data_process_backend.plot_fig
"""Plotly figure object to be initially served to the dash app."""

app = Dash(
    __name__,
    # external_stylesheets=[dbc.themes.BOOTSTRAP, dbc.icons.FONT_AWESOME], #now served locally in the assets folder
    title="Impedance Spectroscopy",
)
"""The dash app variable."""

color_mode_switch = html.Span(
    [
        dbc.Label(
            className="fa fa-moon",
            html_for="switch",
        ),
        dbc.Switch(
            id="switch",
            value=True,
            className="d-inline-block ms-1",
            persistence=True,
        ),
        dbc.Label(
            className="fa fa-sun",
            html_for="switch",
        ),
    ],
)
"""A dash html Span element for a theme switching button."""

offcanvas = html.Div([
    dbc.Offcanvas(
        dcc.Markdown(
            "",
            mathjax=True,
            className="offcanvas-markdown",
        ),
        id="offcanvas",
        title="Impedance Spectroscopy",
        is_open=False,
        class_name="offcanvas-style",
    ),
])
"""A dash html div for a canvas that can be rolled into sight when called. Here theory of the impedance analysis is mentioned."""

save_file_modal = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Save Files as")),
        dbc.ModalBody(
            dbc.Input(
                id="save-as-file",
                placeholder="Filename without extensions...",
                type="text",
                value="impedance",
                pattern=r'[^.\\/:*?"<>|]+',
                # title=r'Filename should not contain .\\/:*?"\'<>|',
            ),
        ),
        dbc.ModalFooter([
            dbc.Button(
                "Save",
                id="confirm-save",
                className="ms-auto",
                n_clicks=0,
                outline=True,
                color="success",
            ),
            dbc.Button(
                "Cancel",
                id="cancel-save",
                className="ms-auto",
                n_clicks=0,
                outline=True,
                color="danger",
            ),
        ]),
    ],
    id="modal",
    is_open=False,
)
"""Dash modal to appear whenever the save button is clicked."""


# mathjax = 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js'
# # app.scripts.append_script({ 'external_url' : mathjax })
app.layout = html.Div(
    [
        dcc.Store(id="check-running", data=False),
        dcc.Interval(id="progress-interval", interval=300, disabled=False),
        dcc.Interval(id="graph-interval-component", interval=300, disabled=False),
        dcc.Interval(id="interval-component", interval=300, disabled=False),
        save_file_modal,
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1(
                            html.A(
                                "HP 4192A Impedance Analyzer",
                                id="open-offcanvas",
                            )
                        ),
                        offcanvas,
                    ],
                    width="auto",
                ),
            ],
            class_name="content-rows",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        color_mode_switch,
                        dbc.Label(
                            "Measurement Mode",
                            class_name="dual-switch-label",
                        ),
                        dbc.RadioItems(
                            options=[
                                {
                                    "label": "Impedance",
                                    "value": 0,
                                },
                                {
                                    "label": "Permittivity",
                                    "value": 1,
                                },
                            ],
                            value=0,
                            id="meas-type-select",
                            inline=True,
                            class_name="dual-switch",
                        ),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText(
                                    "Start frequency : ",
                                ),
                                dbc.Input(
                                    type="number",
                                    step=0.1,
                                    id="start-freq",
                                    value=5.0,
                                    min=5.0,
                                    max=13.0e6,
                                ),
                                dbc.InputGroupText(
                                    "Hz",
                                ),
                            ],
                        ),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText(
                                    "End frequency : ",
                                ),
                                dbc.Input(
                                    type="number",
                                    step=0.1,
                                    id="end-freq",
                                    value=1000.0,
                                    min=5.0,
                                ),
                                dbc.InputGroupText(
                                    "Hz",
                                ),
                            ],
                        ),
                        dbc.Label(
                            "Frequency Sweep",
                            class_name="dual-switch-label",
                        ),
                        dbc.RadioItems(
                            options=[
                                {
                                    "label": "Linear",
                                    "value": 0,
                                },
                                {
                                    "label": "Logarithmic",
                                    "value": 1,
                                },
                            ],
                            value=0,
                            id="freq-sweep-select",
                            inline=True,
                            class_name="dual-switch",
                        ),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText(
                                    "Frequency step : ",
                                ),
                                dbc.Input(
                                    type="number",
                                    step=0.1,
                                    id="freq-step",
                                    value=1.0,
                                    min=0.1,
                                ),
                                dbc.InputGroupText(
                                    "Hz",
                                ),
                            ],
                        ),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText(
                                    "Steps per decade : ",
                                ),
                                dbc.Input(
                                    type="number",
                                    step=1.0,
                                    id="decade-step",
                                    value=1.0,
                                    min=1.0,
                                    disabled=True,
                                ),
                            ],
                        ),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText(
                                    "Diameter : ",
                                ),
                                dbc.Input(
                                    type="number",
                                    step=0.01,
                                    id="sample-diameter",
                                    value=1.0,
                                    min=1e-2,
                                    disabled=True,
                                ),
                                dbc.InputGroupText(
                                    "mm",
                                ),
                            ],
                        ),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText(
                                    "Thickness : ",
                                ),
                                dbc.Input(
                                    type="number",
                                    step=0.01,
                                    id="sample-thickness",
                                    value=1.0,
                                    min=1e-2,
                                    disabled=True,
                                ),
                                dbc.InputGroupText(
                                    "µm",
                                ),
                            ],
                        ),
                        dbc.Switch(
                            id="temperature-switch",
                            label="Sample Temperature",
                            value=False,
                        ),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText(
                                    "Stray Capacitance : ",
                                ),
                                dbc.Input(
                                    type="number",
                                    step=0.0001,
                                    id="stray-capacitance",
                                    value=0.0,
                                    min=0.0,
                                    disabled=True,
                                ),
                                dbc.InputGroupText(
                                    "F",
                                ),
                            ],
                        ),
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText(
                                    "No. of passes : ",
                                ),
                                dbc.Input(
                                    type="number",
                                    step=1,
                                    id="meas-passes",
                                    value=1,
                                    min=1,
                                ),
                            ],
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        "Start",
                                        color="primary",
                                        outline=True,
                                        id="start-click",
                                    ),
                                    className="d-grid gap-2 col-6 mx-auto",
                                ),
                                dbc.Col(
                                    dbc.Button(
                                        "Stop!",
                                        color="warning",
                                        outline=True,
                                        id="stop-click",
                                        disabled=True,
                                    ),
                                    className="d-grid gap-2 col-6 mx-auto",
                                ),
                            ],
                            class_name="button-rows",
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        "Save Plot / Results",
                                        color="success",
                                        outline=True,
                                        id="save-results",
                                        disabled=True,
                                    ),
                                    className="d-grid gap-2 col-6 mx-auto",
                                ),
                            ],
                            class_name="button-rows",
                        ),
                        dbc.Progress(
                            id="progress-bar",
                            value="0",
                            striped=True,
                            animated=True,
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.Button(
                                        "Shutdown App!",
                                        color="danger",
                                        outline=True,
                                        id="shutdown-app",
                                        disabled=False,
                                    ),
                                    className="d-grid gap-2 col-6 mx-auto",
                                ),
                            ],
                            class_name="shutdown-button-rows",
                            align="end",
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dcc.Graph(
                            id="graph",
                            figure=fig,
                            config=CONFIG,
                            mathjax=True,
                            className="dcc-graph",
                        ),
                        dcc.Download(id="download-data"),
                        dcc.Download(id="download-plot"),
                    ],
                    width=8,
                ),
            ],
            class_name="content-rows",
        ),
    ],
    className="app-ui-division",
)

http_server = WSGIServer(
    ("", 8060),
    app.server,
)
"""The WSGI Production server to run the dash app."""

# ********************************************************************************
# ********************************************************************************
# ********************************************************************************
# ********************************************************************************
# ********************************************************************************


@callback(
    Output(
        "sample-diameter",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "sample-thickness",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "stray-capacitance",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "temperature-switch",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "graph",
        "figure",
        allow_duplicate=True,
    ),
    Input("meas-type-select", "value"),
    prevent_initial_call=True,
)
def toggleMeasType(meas_mode: int) -> tuple[bool, bool, bool, bool, go.Figure | Patch]:
    """Method called when the measurement type is toggled.

    Parameters
    ----------
    meas_mode : int
        integer indicating if the measurement is Impedance mode (0) or Permittivity mode (1).

    Returns
    -------
    bool
        Boolean indicating whether the sample-diameter field is to be disabled.
    bool
        Boolean indicating whether the sample-thickness field is to be disabled.
    bool
        Boolean indicating whether the stray-capacitance field is to be disabled.
    bool
        Boolean indicating whether the temperature-switch is to be disabled.
    plotly.graph_objects.Figure | dash.Patch
        Plotly figure object or dash Patch object for the graph division of the dash app.

    """
    temp_figure = data_process_backend.plot_fig
    patched_figure = Patch()
    patched_figure["layout"]["yaxis"]["title"]["text"] = (
        r"$\varepsilon ' $" if meas_mode else r"$\text{Real Impedance } [ \Omega ]$"
    )

    patched_figure["layout"]["yaxis2"]["title"]["text"] = (
        r'$ \varepsilon  "$' if meas_mode else r"$\text{Imaginary Impedance } [ \Omega ]$"
    )

    patched_figure["layout"]["xaxis3"]["title"]["text"] = (
        r"$ \varepsilon '$" if meas_mode else r"$\text{Real Impedance } [ \Omega ]$"
    )

    patched_figure["layout"]["yaxis3"]["title"]["text"] = (
        r'$ \varepsilon "$' if meas_mode else r"$\text{Imaginary Impedance } [ \Omega ]$"
    )

    temp_figure.update_layout(
        {
            "yaxis": {
                "title": {
                    "text": r"$ \varepsilon '$" if meas_mode else r"$\text{Real Impedance } [ \Omega ]$",
                }
            },
            "yaxis2": {
                "title": {
                    "text": r'$ \varepsilon " $' if meas_mode else r"$\text{Imaginary Impedance } [ \Omega ]$",
                }
            },
            "xaxis3": {
                "title": {
                    "text": r"$ \varepsilon '$" if meas_mode else r"$\text{Real Impedance } [ \Omega ]$",
                }
            },
            "yaxis3": {
                "title": {
                    "text": r'$ \varepsilon "$' if meas_mode else r"$\text{Imaginary Impedance } [ \Omega ]$",
                }
            },
        },
        overwrite=False,
    )

    data_process_backend.plot_fig = temp_figure

    return (
        False if meas_mode else True,
        False if meas_mode else True,
        False if meas_mode else True,
        False if meas_mode else True,
        patched_figure,
    )


# ******************************************************************************


@callback(
    Output(
        "freq-step",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "decade-step",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "graph",
        "figure",
        allow_duplicate=True,
    ),
    Input("freq-sweep-select", "value"),
    prevent_initial_call=True,
)
def toggleFrequencySweepMode(sweep_mode: int) -> tuple[bool, bool, go.Figure | Patch]:
    """Method called when the frequency sweep mode is toggled.

    Parameters
    ----------
    sweep_mode : int
        Integer indicating if the frequency sweeep is in linear (0) or logarithmic (1) mode.

    Returns
    -------
    bool
        Boolean indicating whether the freq-step field is to be disabled.
    bool
        Boolean indicating whether the decade-step field is to be disabled.
    plotly.graph_objects.Figure | dash.Patch
        The plotly figure od dash Patch object to be sent to the graph division of the dash app.

    """
    patched_figure = Patch()
    temp_figure = data_process_backend.plot_fig

    patched_figure["layout"]["xaxis"]["type"] = "log" if sweep_mode else "linear"
    patched_figure["layout"]["xaxis2"]["type"] = "log" if sweep_mode else "linear"

    temp_figure.update_layout(
        {
            "xaxis": {"type": "log" if sweep_mode else "linear"},
            "xaxis2": {"type": "log" if sweep_mode else "linear"},
        },
        overwrite=False,
    )

    data_process_backend.plot_fig = temp_figure

    if sweep_mode:
        return True, False, patched_figure
    else:
        return False, True, patched_figure


# *********************************************************************************


@callback(
    Output("offcanvas", "is_open"),
    Input("open-offcanvas", "n_clicks"),
    State("offcanvas", "is_open"),
    prevent_initial_call=True,
)
def toggleOffcanvas(n1: int, is_open: bool) -> bool:
    """Method called when the title is clicked. It slides in a panel with information.

    Parameters
    ----------
    n1 : int
        Integer indicating the number of clicks on the open-offcanvas element.
    is_open : bool
        Boolean indicating if the offcanvas element is open.

    Returns
    -------
    bool
        Boolean indicating if the offcanvas element is to be opened.

    """
    if n1:
        return not is_open
    return is_open


# ********************************************************************************


clientside_callback(
    ClientsideFunction("clientside", "theme_switched"),
    Output("switch", "id"),
    Input("switch", "value"),
    # prevent_initial_call=True,
)
"""Method called on the clientside when the theme switch is toggled."""

# ********************************************************************************


@callback(
    Output("graph", "figure"),
    Input("switch", "value"),
    # prevent_initial_call=True,
)
def updateFigureTemplate(switch_on: int) -> go.Figure | Patch:
    """Method called to change the theme of the figure when the theme of the app is changed.

    Parameters
    ----------
    switch_on : int
        Integer indicating the state of the switch. 0 is OFF and 1 is ON.

    Returns
    -------
    plotly.graph_objects.Figure | dash.Patch
        The plotly figure or dash Patch object to be sent to the graph division of the app.

    """

    # When using Patch() to update the figure template, you must use the figure template dict
    # from plotly.io  and not just the template name
    # template = pio.templates["bootstrap"] if switch_on else pio.templates["bootstrap_dark"]

    template = BOOTSTRAP if switch_on else BOOTSTRAP_DARK

    temp_figure = data_process_backend.plot_fig
    patched_figure = Patch()
    patched_figure["layout"]["template"] = template

    temp_figure.update_layout(dict(template=template), overwrite=False)

    data_process_backend.plot_fig = temp_figure

    return patched_figure


# ********************************************************************************


@callback(
    Input("stop-click", "n_clicks"),
    prevent_initial_call=True,
)
def cancelOrStopMeasurement(n: int) -> None:
    """Method called when the Stop button is clicked on the app.

    Parameters
    ----------
    n : int
        Integer indicating the number of clicks of the stop-click button.

    """
    data_process_backend.meas_cancelled = True


# ***********************************************************************************


@callback(
    Output(
        "check-running",
        "data",
        allow_duplicate=True,
    ),
    Output(
        "interval-component",
        "disabled",
        allow_duplicate=True,
    ),
    Input("start-click", "n_clicks"),
    State("meas-type-select", "value"),
    State("start-freq", "value"),
    State("end-freq", "value"),
    State("freq-sweep-select", "value"),
    State("freq-step", "value"),
    State("decade-step", "value"),
    State("sample-diameter", "value"),
    State("sample-thickness", "value"),
    State("stray-capacitance", "value"),
    State("temperature-switch", "value"),
    State("meas-passes", "value"),
    # background=True,
    prevent_initial_call=True,
)
def startMeasurement(
    n_clicks: int,
    meas_type: int,
    start_freq: float,
    end_freq: float,
    sweep_type: int,
    sweep_step: float,
    steps_decade: int,
    sample_d: float,
    sample_t: float,
    stray_Farad: float,
    get_temperature: bool,
    pass_no: int,
) -> tuple[bool, bool]:
    """Method called when the Start button is clicked.

    Parameters
    ----------
    n_clicks : int
        Integer indicating the number of times the Start button was clicked.
    meas_type : int
        integer indicating the measurement mode. 0 is Impedance and 1 is Permittivity.
    start_freq : float
            The starting frequency for the measurement.
    end_freq : float
        The ending frequency for the measurement.
    sweep_type : int
        Integer indicating the sweep mode of the measurement. 0 is linear and 1 is logarithmic.
    sweep_step : float
        The frequency steps in case of a linear sweep.
    steps_decade : int
        Integer indicating the number of steps per decade in case of a logarithmic sweep.
    sample_d : float
        The diameter of the sample (PVAC) in mm.
    sample_t : float
        The thickness of the sample (PVAC) in µm.
    stray_Farad : float
        The stray capacitance value to be subtracted in farads.
    get_temperature : bool
        Boolean indicating if the sample temperature is to be recorded.
    pass_no : int
        Number of measurement passes required.

    Returns
    -------
    bool
        Boolean data stored in the check-running store.
    bool
        Boolean indicating if the interval-component interval is to be disabled.

    """
    logarithmic = False
    if sweep_type:
        logarithmic = True
    if not meas_type:
        # Impedance measurement mode.
        data_process_backend.startMeasurementThread(
            start_freq,
            end_freq,
            sweep_step,
            logarithmic,
            steps_decade,
            sample_d,
            sample_t,
            stray_Farad,
            get_temperature,
            pass_no,
        )
    else:
        # Permittivity measurement mode
        data_process_backend.startMeasurementThread(
            start_freq,
            end_freq,
            sweep_step,
            logarithmic,
            steps_decade,
            sample_d,
            sample_t,
            stray_Farad,
            get_temperature,
            pass_no,
            False,
        )

    return True, False


# ********************************************************************************


@callback(
    Output(
        "start-click",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "stop-click",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "shutdown-app",
        "disabled",
        allow_duplicate=True,
    ),
    Output("meas-type-select", "options"),
    Output("freq-sweep-select", "options"),
    Output("start-freq", "disabled"),
    Output("end-freq", "disabled"),
    Output(
        "freq-step",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "decade-step",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "sample-diameter",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "sample-thickness",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "stray-capacitance",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "temperature-switch",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "meas-passes",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "interval-component",
        "disabled",
        allow_duplicate=True,
    ),
    Output(
        "save-results",
        "disabled",
        allow_duplicate=True,
    ),
    Input("check-running", "data"),
    State("meas-type-select", "value"),
    State("meas-type-select", "options"),
    State("freq-sweep-select", "value"),
    State("freq-sweep-select", "options"),
    prevent_initial_call=True,
)
def checkRunningProgress(
    running: bool,
    meas_mode: int,
    meas_type_options: dict,
    sweep_mode: int,
    sweep_type_options: dict,
) -> tuple[
    bool,
    bool,
    bool,
    list,
    list,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
    bool,
]:
    """Method called when the check-running interval fires.

    Parameters
    ----------
    running : bool
        Boolean indicating if the measurement is running.
    meas_mode : int
        Integer indicating the measurement type. 0 is Impedance and 1 is Permittivity mode.
    meas_type_options : dict
        Dictionary with options to the meas-type elements. Used to enable or disable the radio items for the meas mode selection.
    sweep_mode : int
        Integer indicating the sweep mode for the measurement. 0 is linear and 1 is logarithmic.
    sweep_type_options : dict
        Dictionary with options to the sweep type elements. Used to enable or disable the radio items for the sweep type selection.

    Returns
    -------
    bool
        Boolean indicating if the start-click button is disabled.
    bool
        Booleann indicating if the stop-click button is disabled.
    bool
        Boolean indicating if the shutdown-app button is disabled.
    dict
        Dictionary with options to the meas-type elements. Used to enable or disable the radio items for the meas mode selection.
    dict
        Dictionary with options to the sweep-type elements. Used to enable or disable the radio items for the sweep type selection.
    bool
        Boolean indicating if the start-freq field is disabled.
    bool
        Boolean indicating if the end-freq field is disabled.
    bool
        Boolean indicating if the freq-step field is disabled.
    bool
        Booleann indicating if the decade-step field is disabled.
    bool
        Boolean indicating if the sample-diameter field is disabled.
    bool
        Boolean indicating if the sample-thickness field is disabled.
    bool
        Boolean indicating if the stray-capacitance field is disabled.
    bool
        Boolean indicating if the temperature-switch is disabled.
    bool
        Booleann indicating if the meas-passes field is disabled.
    bool
        Boolean indicating if the interval-component interval is disabled.
    bool
        Boolean indicating if the save-results button is disabled.

    """
    meas_temp = list(meas_type_options)
    sweep_temp = list(sweep_type_options)
    for item in meas_temp:
        item["disabled"] = data_process_backend.meas_running
    for item in sweep_temp:
        item["disabled"] = data_process_backend.meas_running
    if running:
        return (
            True,  # disable start-click button
            False,  # disable stop-click button
            True,  # disable shutdown app button
            meas_temp,  # meas-type-select-options
            sweep_temp,  # freq-sweep-select options
            True,  # disable start-freq
            True,  # disable end-freq
            True,  # disable freq-step
            True,  # disable decade-step
            True,  # disable sample-diameter
            True,  # disable sample-thickness
            True,  # disable stray-capacitance
            True,  # disable temperature-switch
            True,  # disable meas-passes
            False,  # disable graph data process interval
            True,  # disable download button
        )
    else:
        return (
            False,  # disable start-click button
            True,  # disable stop-click button
            False,  # disable shutdown app button
            meas_temp,  # meas-type-select-options
            sweep_temp,  # freq-sweep-select options
            False,  # disable start-freq
            False,  # disable end-freq
            True if sweep_mode else False,  # disable freq-step
            False if sweep_mode else True,  # disable decade-step
            False if meas_mode else True,  # disable sample-diameter
            False if meas_mode else True,  # disable sample-thickness
            False if meas_mode else True,  # disable stray-capacitance
            False if meas_mode else True,  # disable temperature-switch
            False,  # disable meas-passes
            True if data_process_backend.meas_completed else False,  # disabled graph data process interval
            False if data_process_backend.meas_completed else True,  # disable the download button
        )


# ********************************************************************************


@callback(
    Output("progress-bar", "value"),
    Output("check-running", "data"),
    Input("progress-interval", "n_intervals"),
    prevent_initial_call=True,
)
def updateProgress(n: int) -> tuple[int, bool]:
    """Method called by the progress-interval interval component on firing.

    Parameters
    ----------
    n : int
        Integer indicating the number of intervals the interval component was fired.

    Returns
    -------
    int
        Integer for the current value of the progress bar.
    bool
        Boolean indicating whether the measurement is running.

    """
    return data_process_backend.progress_current, data_process_backend.meas_running


# ********************************************************************************


@callback(
    Output(
        "graph",
        "figure",
        allow_duplicate=True,
    ),
    Output(
        "interval-component",
        "disabled",
        allow_duplicate=True,
    ),
    Input("interval-component", "n_intervals"),
    prevent_initial_call=True,
)
def updateGraphLive(n: int) -> tuple[go.Figure | Patch, bool]:
    r"""Method that is called in periodic intervals.

    Parameters
    ----------
    n : int
        The iteration of the update interval.

    Returns
    -------
    plotly.graph_objects.Figure | dash.Patch
        The plotly figure or dash Patch object to be sent to the graph division of the app.
    bool
        Boolean indicating if the interval-component interval is to be disabled.

    """
    patched_figure = Patch()
    data = data_process_backend.plot_fig.to_dict()["data"]
    patched_figure["data"] = data

    return patched_figure, not data_process_backend.meas_running


# ********************************************************************************


@callback(
    Output(
        "modal",
        "is_open",
        allow_duplicate=True,
    ),
    Output("save-as-file", "value"),
    Input("save-results", "n_clicks"),
    Input("cancel-save", "n_clicks"),
    State("modal", "is_open"),
    State("meas-type-select", "value"),
    prevent_initial_call=True,
)
def confirmFileSaveModalOpen(n1: int, n2: int, is_open: bool, meas_type: int) -> tuple[bool, str]:
    """Method called when the Save results button is clicked. It will open a Modal to confirm the file save.

    Parameters
    ----------
    n1 : int
        Integer holding the no. of clicks on the Save button.
    n2 : int
        Integer holding the number of clicks on the Cancel button in the Modal.
    is_open : bool
        Boolean indicating if the Modal is currently open.
    meas_type : int
        Integer indicating the type of measurement conducted. 0 is impedance and 1 is permittivity.

    Returns
    -------
    bool
        Boolean indicating if the Modal is to be opened.
    """
    filename_noextensions = "impedance"
    if meas_type:
        filename_noextensions = "permittivity"
    if n1 or n2:
        return not is_open, filename_noextensions
    return is_open, filename_noextensions


# *********************************************************************************


@callback(
    Output("confirm-save", "disabled"),
    Output("save-as-file", "valid"),
    Output("save-as-file", "invalid"),
    Input("save-as-file", "value"),
    prevent_initial_call=True,
)
def checkFilenameValidity(filename: str) -> tuple[bool, bool, bool]:
    """Method to check filename validity.

    Parameters
    ----------
    filename : str
        Name of the file to be saved as.

    Returns
    -------
    tuple[bool, bool, bool]
        Boolean to disabled the save button in the modal, enable valid property of the input field, enable invalid property of the input field respectively.
    """
    pattern = r'[^.\\/:*?"\'<>|]+'
    match = re.fullmatch(pattern, filename)
    if match is not None:
        return False, True, False
    else:
        return True, False, True


# *********************************************************************************


@callback(
    Output(
        "modal",
        "is_open",
        allow_duplicate=True,
    ),
    Input("confirm-save", "n_clicks"),
    State("graph", "figure"),
    State("save-as-file", "value"),
    prevent_initial_call=True,
)
def saveDataAndPlot(n_clicks: int, ex_fig: dict, filename: str) -> bool:
    """Method called when Save Results button is clicked.

    Parameters
    ----------
    n_clicks : int
        Integer indicating the number of times the save-results button was clicked.
    ex_fig : dict
        The dictionary with the figure object of the graph area in the Dash app.
    filename : str
        The filename to save the results and plot as.

    Returns
    -------
    bool
        The boolean to close the Modal.
    """
    # fig = data_process_backend.plot_fig
    fig = go.Figure(ex_fig)
    fig.write_html(
        filename + ".html",
        config=CONFIG,
        include_plotlyjs="cdn",
        include_mathjax="cdn",
    )

    data_process_backend.dataframe_data.to_csv(filename + ".csv")
    return False


# *********************************************************************************


@callback(
    Input("shutdown-app", "n_clicks"),
    prevent_initial_call=True,
)
def shutdownApp(n: int) -> None:
    """Method called when the shutdown-app button is clicked.

    Parameters
    ----------
    n : int
        Integer indicating the number of times the shutdown-app button was clicked.

    """
    data_process_backend.prepareForShutdown()


# *********************************************************************************


if __name__ == "__main__":
    # app.run(debug=True, port="8060")
    # impedance_spec_app.http_server.serve_forever()
    waitress.serve(app.server, host="localhost", port="8060", expose_tracebacks=True, threads=8)
