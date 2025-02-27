Impedance Spectroscopy Dash
===========================
What is Impedance Spectroscopy Dash?
------------------------------------
It is a python program with a frontend designed using plotly Dash. This was specifically written for the Impedance Spectroscopy Lab course provided to students at the Ulm University in Germany.

What does it do?
----------------
The program communicates with an HP 4192A Impedance Spectrometer via GPIB to record the data. The lab course specifically focussed on an RC parallel circuit and a PVAC sample. For the PVAC sample, the temperature was also recorded.

Requirements
------------
This program was written with a Windows 10/11 system in mind. Install the necessary libraries mentioned in the requirements file. For recording the temperature, a Voltcraft VC870 multimeter was used. Readings from the multimeter are recorded on the PC via USB, using the sigrok-cli. Refer to https://sigrok.org/wiki/Downloads. The **sigrok-cli** folder needs to be placed in the **src** folder. Make sure to edit the `sigrok_cmd` in the `Databackend.multimeterGetTemperature` method with the specific voltmeter driver and location (hid, serial etc.) that is applicable to you.

Note of Thanks
--------------
Big thanks to all the authors of the different libraries that I had to use.

Frontend Example
----------------
![image](https://github.com/user-attachments/assets/3316523f-8e04-44c9-9e0b-6844f9e22b0e)

