import math
import re
import pyvisa
import time


class HP4192A:
    """Class representing the HP 4192A GPIB device object.

    Methods
    -------
    __init__:
        Method called when class is initialized.
    isConnected:
        Method that returns the connected boolean of the class.
    closeConnection:
        Method called to close the connection to the device.
    sendCommandString:
        Method called to generate and send a command string to the HP 4192A device.
    readFromInstrument:
        Method used to read data from the HP 4192A device.

    """

    FUNCTION_A = [f"A{i}" for i in range(1, 5)]
    """String constants for Function A.
    
    A1 : Z/Y
    A2 : R/G
    A3 : L
    A4 : C
    Refer to HP 4192A manual for more information.
    https://www.keysight.com/us/en/assets/9018-05094/user-manuals/9018-05094.pdf
    """
    FUNCTION_B = [f"B{i}" for i in range(1, 4)]
    """String constants for Function B.

    B1 : DEG Q
    B2 : RAD D
    B3 : R/G
    Refer to HP 4192A manual for more information.
    https://www.keysight.com/us/en/assets/9018-05094/user-manuals/9018-05094.pdf
    """
    CIRCUIT_MODE = [f"C{i}" for i in range(1, 4)]
    """String constants for circuit mode.
    
    C1 : Auto
    C2 : Series
    C3 : Parallel
    Refer to HP 4192A manual for more information.
    https://www.keysight.com/us/en/assets/9018-05094/user-manuals/9018-05094.pdf
    """
    DATA_READY = ["D0", "D1"]
    """Data ready command.

    D0 : OFF
    D1 : ON
    Refer to HP 4192A manual for more information.
    https://www.keysight.com/us/en/assets/9018-05094/user-manuals/9018-05094.pdf
    """
    ZY_RANGE = [f"R{i}" for i in range(1, 9)]
    """Range level.

    R8 is AUTO
    Refer to HP 4192A manual for more information.
    https://www.keysight.com/us/en/assets/9018-05094/user-manuals/9018-05094.pdf
    """

    HIGH_SPEED = ["H0", "H1"]
    """High Speed.

    H0 is OFF. H1 is ON.
    Refer to HP 4192A manual for more information.
    https://www.keysight.com/us/en/assets/9018-05094/user-manuals/9018-05094.pdf
    """

    TRIGGER = [f"T{i}" for i in range(1, 4)]
    """Trigger mode for the instrument.

    T1 is internal.
    Refer to HP 4192A manual for more information.
    https://www.keysight.com/us/en/assets/9018-05094/user-manuals/9018-05094.pdf
    """
    OP_DATA_FORMAT = ["F0", "F1"]
    """Output data format.
    
    F0 : Displays A/B
    F1 : Displays A/B/C
    Refer to HP 4192A manual for more information.
    https://www.keysight.com/us/en/assets/9018-05094/user-manuals/9018-05094.pdf
    """
    EXECUTE = ["EX"]
    """Command code to trigger the instrument.

    Refer to HP 4192A manual for more information.
    https://www.keysight.com/us/en/assets/9018-05094/user-manuals/9018-05094.pdf
    """
    AVERAGE = ["V0", "V1"]
    """Average data.
    
    V0 : OFF
    V1 : ON
    Refer to HP 4192A manual for more information.
    https://www.keysight.com/us/en/assets/9018-05094/user-manuals/9018-05094.pdf
    """
    BIAS = ["BI", "TB", "PB"]
    """Measurement bias.
    
    BI : SPOT BIAS
    TB : START BIAS
    PB : STOP BIAS
    Refer to HP 4192A manual for more information.
    https://www.keysight.com/us/en/assets/9018-05094/user-manuals/9018-05094.pdf
    """

    FREQUENCY_SETT = ["FR", "TF", "PF"]
    """Frequency setting.
    
    FR : SPOT FREQ
    TF : START FREQ
    PF : STOP FREQ
    Refer to HP 4192A manual for more information.
    https://www.keysight.com/us/en/assets/9018-05094/user-manuals/9018-05094.pdf
    """

    __rm: pyvisa.ResourceManager
    """VISA resource manager"""

    ___gpib_address: str
    """The GPIB address of the device."""

    __hp_4192a: pyvisa.resources.Resource
    """The HP 4192A at the GPIB address."""

    __device_connected: bool = False
    """Boolean indicating if the device connection was succesful."""

    current_range: str = ""
    """Current range deduced from the raw string received from the HP 4192A."""

    def __init__(self, gpib="Auto") -> None:
        """Method called when class is initialized.

        Parameters
        ----------
        gpib : str, optional
            The GPIB address of the device, by default "Auto"
        """
        self.__rm = pyvisa.ResourceManager()

        if gpib in "Auto":
            for address in self.__rm.list_resources():
                if "GPIB" in address:
                    self.__gpib_address = address
                    break
        else:
            if gpib in self.__rm.list_resources():
                self.__gpib_address = gpib
            else:
                print("Entered GPIB address does not exist.")
                return

        try:
            self.__hp_4192a = self.__rm.open_resource(self.__gpib_address)
            self.__hp_4192a.timeout = None
            self.__hp_4192a.read_termination = "\r\n"
            """Refer to HP 4192A manual for more information.
            https://www.keysight.com/us/en/assets/9018-05094/user-manuals/9018-05094.pdf"""
            self.__device_connected = True
        except Exception as e:
            print(e)
        pass

    def isConnected(self) -> bool:
        """Method that returns the connected boolean of the class.

        Returns
        -------
        bool
            Boolean with the connection status of the device.
        """
        return self.__device_connected

    def closeConnection(self) -> None:
        """Method called to close the connection to the device."""
        if self.__device_connected:
            self.__hp_4192a.close()
            self.__rm.close()

    def sendCommandString(
        self,
        data_ready: int | None = None,
        zy_range: int | None = None,
        high_speed: int | None = None,
        average: int | None = None,
        output_format: int | None = None,
        function_A: int | None = None,
        function_B: int | None = None,
        circuit_mode: int | None = None,
        trigger: int | None = None,
        osc_level: float | None = None,
        test_signal_freq: float | None = None,
        dc_bias: bool = True,
        bias_setting: float | None = None,
        execute: bool = False,
        sleep_adder: int = 100,
    ) -> None:
        """Method called to generate and send a command string to the HP 4192A device.

        Parameters
        ----------
        data_ready : int | None, optional
            Index for the DATA_READY parameter, by default None
        zy_range : int | None, optional
            Index for the Z/Y range level parameter, by default None
        average : int | None, optional
            Index for the averaging parameter, by default None
        output_format : int | None, optional
            Index for the output format parameter, by default None
        function_A : int | None, optional
            Index for the Function A parameter, by default None
        function_B : int | None, optional
            Index for the Function B parameter, by default None
        circuit_mode : int | None, optional
            Index for the circuit mode parameter, by default None
        trigger : int | None, optional
            Index for the trigger mode parameter, by default None
        osc_level : float | None, optional
            Value for the oscillation level of the test frequency signal, by default None
        test_signal_freq : float | None, optional
            Value for the frequency of the test signal, by default None
        dc_bias : bool, optional
            Boolean indicating if DC Bias is ON, by default True
        bias_setting : float | None, optional
            Value for the DC Bias if set to ON, by default None
        execute : bool, optional
            Boolean indicating if the execute command is to be added to the commands., by default False.
        sleep_adder : int, optional
            Integer with the sleep to be added per command in ms.
        """
        if self.__device_connected:
            sleeptime = 0
            sleep_adder = sleep_adder
            command_string = ""
            if data_ready is not None:
                command_string = "".join([command_string, self.DATA_READY[data_ready]])
                sleeptime += sleep_adder
            if zy_range is not None:
                command_string = "".join([command_string, self.ZY_RANGE[zy_range]])
                sleeptime += sleep_adder
            if high_speed is not None:
                command_string = "".join([command_string, self.HIGH_SPEED[high_speed]])
                sleeptime += sleep_adder
            if average is not None:
                command_string = "".join([command_string, self.AVERAGE[average]])
                sleeptime += sleep_adder
            if output_format is not None:
                command_string = "".join([command_string, self.OP_DATA_FORMAT[output_format]])
                sleeptime += sleep_adder
            if function_A is not None:
                command_string = "".join([command_string, self.FUNCTION_A[function_A]])
                sleeptime += sleep_adder
            if function_B is not None:
                command_string = "".join([command_string, self.FUNCTION_B[function_B]])
                sleeptime += sleep_adder
            if circuit_mode is not None:
                command_string = "".join([command_string, self.CIRCUIT_MODE[circuit_mode]])
                sleeptime += sleep_adder
            if trigger is not None:
                command_string = "".join([command_string, self.TRIGGER[trigger]])
                sleeptime += sleep_adder
            if osc_level is not None:
                command_string = "".join([command_string, f"OL{osc_level:.3f}EN"])
                sleeptime += sleep_adder
            if test_signal_freq is not None:
                freq_kHz = test_signal_freq / 1000
                freq_cmd: str
                if freq_kHz < 10.0:
                    freq_cmd = f"{(freq_kHz):09.6f}"
                elif freq_kHz < 100.0:
                    freq_cmd = f"{(freq_kHz):09.5f}"
                elif freq_kHz < 1000.0:
                    freq_cmd = f"{(freq_kHz):09.4f}"
                else:
                    freq_cmd = f"{(freq_kHz):09.3f}"
                command_string = "".join([command_string, f"FR{freq_cmd}EN"])
                sleeptime += sleep_adder
            if dc_bias:
                if bias_setting is not None:
                    command_string = "".join([command_string, f"BI{bias_setting:.2f}EN"])
                    sleeptime += sleep_adder
            else:
                command_string = "".join([command_string, "I0"])  # Refer to HP 4192A manual for more information.
                # https://www.keysight.com/us/en/assets/9018-05094/user-manuals/9018-05094.pdf
                sleeptime += sleep_adder
            if execute:
                command_string = "".join([command_string, "EX"])
                sleeptime += sleep_adder

            self.__hp_4192a.write(command_string)
            time.sleep(sleeptime / 1000)

    def readFromInstrument(self) -> list:
        """Method used to read data from the HP 4192A device.

        Returns
        -------
        list
            list of float values corresponding to display A and display B of the device.
        """
        if self.__device_connected:
            received_string = self.__hp_4192a.read()

            # Regular expression to capture floating numbers in scientific notation
            pattern = r"[-+]?\d*\.\d+E[-+]?\d+"

            # Find all matches
            matches = re.findall(pattern, received_string)

            # Convert matches to floats
            float_numbers = [float(match) for match in matches]
            return float_numbers
        else:
            return [math.nan, math.nan]
