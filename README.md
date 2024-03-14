# BRAIN Installation Instructions

## 1. Pre-installation
Please make sure that you have installed *git* and *MATLAB* on your computer.
### How to install 'git'?
- *git* installer can be found from this link, [git-installer](https://git-scm.com/).
- or, it can be installed via Windows PowerShell using *winget*, a package manager that is already available on Windows 11 and modern versions of Windows 10.
``` PowerShell
winget install --id Git.Git -e --source winget
```
### MATLAB installation and dependent Toolboxes
- *MATLAB* can be installed via its [official website](https://uk.mathworks.com/products/matlab.html)
- Once *MATLAB* is installed, add these Toolboxes into your *MATLAB*:
- [ ] Data Aacquisition Toolbox,
- [ ] Instrument Control Toolbox,
- [ ] Phased Array System Toolbox.
**NOTE**: MATLAB Toolboxes can be added from the 'Add-Ons' Button, which is shown on the *HOME* Page of the *MATLAB* interface.

## 2. BRAIN Installation

### 2.1 Download BRAIN
- *BRAIN* software can be found from the *brain1* repository in the [Bristol UNDT GitHub Webpage](https://github.com/ndtatbristol/brain1),
- In the Bristol UNDT GitHub webpage, find the green 'Code' button and copy the HTTPS link,
- In your computer, choose a local directory where you would like to put the BRIAN software, for example, *C:\Users\xxx\Documents*.
- Run Windows PowerShell in that directory, and input following command to download the BRAIN software:
  ``` PowerShell
  git clone https://github.com/ultrasunix/total-focusing-method-2d-python-example.git
  ```
By far, the *BRAIN* software is downloaded.

### 2.2 Add BRAIN to your MATLAB startup.m
startup.m executes user-specified commands when starting MATLAB, which can enable BRAIN by daulft by adding the BARIN path into the startup.m file.
- Open *MATLAB*, and create a new blank script and named as startup.m
- add following command into the startup.m
  ``` MATLAB
  addpath(genpath('C:\Users\xxx\Documents\brain1'));
  ```
- save the startup.m and restart MATLAB.
By far, *BRAIN* is installed.

## 3. Connect the PeakNDT MicroPulse System to Computer
- connect the PeakNDT MicroPulse System and your computer via an Ethernet cable,
- open **Control Panel**, click **Network Setting Centre**, click **Network and Sharing Centre**, click **Change Adapter Settings** shown on the left
- Right click the ethernet icon which is for the PeakNDT MicroPulse System connection, then click the *Properties*,
- In the **Networking** panel, find the **Internet Protocol Version 4 (TCP/IPv4)** option,
- Choose **Use the following IP address:**, and change the **IP address** to 10.1.1.1,
- Click **OK** to save all the changes.
By far, BRAIN software is installed in the computer and can be activated by typing **brain** in the MATLAB command window.
