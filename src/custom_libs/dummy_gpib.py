import math
import re
import numpy as np
# import pyvisa


class HP4192A:
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

    # __rm: pyvisa.ResourceManager
    """VISA resource manager"""

    ___gpib_address: str
    """The GPIB address of the device."""

    # __hp_4192a: pyvisa.resources.Resource
    """The HP 4192A at the GPIB address."""

    __device_connected: bool = False
    """Boolean indicating if the device connection was succesful."""

    def __init__(self, gpib="Auto") -> None:
        # self.__rm = pyvisa.ResourceManager()

        # if gpib in "Auto":
        #     for address in self.__rm.list_resources():
        #         if "GPIB" in address:
        #             self.__gpib_address = address
        #             break
        # else:
        #     if gpib in self.__rm.list_resources():
        #         self.__gpib_address = gpib
        #     else:
        #         print("Entered GPIB address does not exist.")
        #         return

        try:
            # self.__hp_4192a = self.__rm.open_resource(self.__gpib_address)
            self.__device_connected = True
        except Exception as e:
            print(e)
        pass

    def isConnected(self):
        return self.__device_connected

    def closeConnection(self) -> None:
        # if self.__device_connected:
        #     self.__rm.close()
        return

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
    ):
        if self.__device_connected:
            command_string = ""
            if data_ready is not None:
                command_string = "".join([command_string, self.DATA_READY[data_ready]])
            if zy_range is not None:
                command_string = "".join([command_string, self.ZY_RANGE[zy_range]])
            if average is not None:
                command_string = "".join([command_string, self.AVERAGE[average]])
            if output_format is not None:
                command_string = "".join([command_string, self.OP_DATA_FORMAT[output_format]])
            if function_A is not None:
                command_string = "".join([command_string, self.FUNCTION_A[function_A]])
            if function_B is not None:
                command_string = "".join([command_string, self.FUNCTION_B[function_B]])
            if circuit_mode is not None:
                command_string = "".join([command_string, self.CIRCUIT_MODE[circuit_mode]])
            if trigger is not None:
                command_string = "".join([command_string, self.TRIGGER[trigger]])
            if osc_level is not None:
                command_string = "".join([command_string, f"OL{osc_level:.3f}EN"])
            if test_signal_freq is not None:
                command_string = "".join([command_string, f"FR{(test_signal_freq / 1000):.4f}EN"])
            if dc_bias:
                if bias_setting is not None:
                    command_string = "".join([command_string, f"BI{bias_setting:.2f}EN"])
            else:
                command_string = "".join([command_string, "I0"])  # Refer to HP 4192A manual for more information.

                # https://www.keysight.com/us/en/assets/9018-05094/user-manuals/9018-05094.pdf
            if execute:
                command_string = "".join([command_string, "EX"])

            # self.__hp_4192a.write(command_string)

    def readFromInstrument(self):
        if self.__device_connected:
            received_string = "NGFN-00.000E-06,NBFN+00.648E-06"

            # Regular expression to capture floating numbers in scientific notation
            pattern = r"[-+]?\d*\.\d+E[-+]?\d+"

            # Find all matches
            matches = re.findall(pattern, received_string)

            # Convert matches to floats
            float_numbers = [np.random.randn(), np.random.randn()]
            return float_numbers
        else:
            return [math.nan, math.nan]
