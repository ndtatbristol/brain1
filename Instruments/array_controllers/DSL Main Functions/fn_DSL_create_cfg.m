function res = fn_DSL_create_cfg(varargin)
%USAGE
%   res = fn_ag_disconnect(filename, exp_data, ph_velocity, echo_on)
%INPUTS
%   filename - the filename of the cfg file (also filename of save file)
%   exp_data - exp_data structure
%   ph_velocity - material velocity
%   test_options - test_options structure
%   echo_on - echos information to screen
%OUTPUTS
%   res - successful (1) or unsuccessful (0)
%NOTES
%   create setup file using parameters specified
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%default values
if nargin < 5
    res = 0;
    disp('Not all arguements present');
    return
else
    filename = varargin{1};
    exp_data = varargin{2};
    ph_velocity = varargin{3};
    test_options = varargin{4};
    echo_on = varargin{5};
end;

%calculated values
number_of_elements = length(test_options.tx_ch);
x_element_pitch = ((exp_data.array.el_xc(2) - exp_data.array.el_xc(1))*1e3);
y_width = ((exp_data.array.el_y2(1) - exp_data.array.el_yc(1))*1e3*2);

test_options.sample_freq = str2double(test_options.sample_freq)*1e6;

%Open file
fileID = fopen(filename,'w');

if fileID == -1  %if file couldn't be opened
    res = 0;
    return;
end;

%Write out File Information
fprintf(fileID,'[File Information]\r\n');
fprintf(fileID,'FileType = "FIToolBox"\r\n');
CurrentDateTime = clock;
fprintf(fileID,'Date = "%2d/%2d/%2d"\r\n', CurrentDateTime(3), CurrentDateTime(2), CurrentDateTime(1));
fprintf(fileID,'Time = "%2d:%2d:%2d"\r\n\r\n', CurrentDateTime(4), CurrentDateTime(5), round(CurrentDateTime(6)));

%Write out 2D Array Geometry - assumes 1D array
fprintf(fileID,'[2D Array Geometry]\r\n');
fprintf(fileID,'Frequency (MHz) = %.5f\r\n', (exp_data.array.centre_freq/1e6));
fprintf(fileID,'NumXelements = %d\r\n', number_of_elements);
fprintf(fileID,'ElementXpitch (mm) = %.5f\r\n', x_element_pitch);
fprintf(fileID,'ElementXwidth (mm) = %.5f\r\n', ((exp_data.array.el_x1(1) - exp_data.array.el_xc(1))*1e3*2));
fprintf(fileID,'ArrayXcurvature (mm) = 0.00000\r\n'); %Can be calculated from Bristol Format but not allowing for this at the moment
fprintf(fileID,'ArrayXorientation (deg) = 0.00000\r\n'); %Can be calculated from Bristol Format but not allowing for this at the moment
fprintf(fileID,'NumYelements = 1\r\n'); %can be calculated but assume linear for now
fprintf(fileID,'ElementYpitch (mm) = %.5f\r\n', ((exp_data.array.el_yc(2) - exp_data.array.el_yc(1))*1e3));
fprintf(fileID,'ElementYwidth (mm) = %.5f\r\n', y_width);
fprintf(fileID,'ArrayYcurvature (mm) = 0.00000\r\n'); %Can be calculated from Bristol Format but not allowing for this at the moment
fprintf(fileID,'ArrayYorientation (deg) = 0.00000\r\n'); %Can be calculated from Bristol Format but not allowing for this at the moment
fprintf(fileID,'Array Range Offset (mm) = 0.00000\r\n');
fprintf(fileID,'Calibration File = "<Not A Path>"\r\n\r\n');

%Write out Wedge Geometry - Contact arrays only, values probably not needed
%but calculated for completeness
fprintf(fileID,'[Wedge Geometry]\r\n');
fprintf(fileID,'Name = "Direct coupled - no delay line"\r\n');
fprintf(fileID,'Wedge Angle (deg) = 0.00000\r\n');
fprintf(fileID,'Roof Angle (deg) = 0.00000\r\n');
fprintf(fileID,'Velocity (m/s) = %.5f\r\n', ph_velocity);
fprintf(fileID,'Length (mm) = %.5f\r\n', (number_of_elements * x_element_pitch * 10));
fprintf(fileID,'Width (mm) = %.5f\r\n', (y_width * 10));
fprintf(fileID,'Height (mm) = 0.00000\r\n');
fprintf(fileID,'Elem1 Pos X = %.5f\r\n', exp_data.array.el_xc(1));
fprintf(fileID,'Elem1 Pos Y = %.5f\r\n', exp_data.array.el_yc(1));
fprintf(fileID,'Elem1 Pos Z = %.5f\r\n', exp_data.array.el_zc(1));
fprintf(fileID,'Wedge Pos X = %.5f\r\n', exp_data.array.el_xc(round(number_of_elements / 2)));
fprintf(fileID,'Wedge Pos Y = %.5f\r\n', exp_data.array.el_yc(round(number_of_elements / 2)));
fprintf(fileID,'Autocentre = TRUE\r\n\r\n');

%Write out Beamformer - majority of no consequence to FMC and are hard
%coded to their default values
fprintf(fileID,'[Beamformer]\r\n');
fprintf(fileID,'Tx Skew (deg) = 0.000000\r\n');
fprintf(fileID,'Tx Width (deg) = 30.00000\r\n');
fprintf(fileID,'Rx Skew (deg) = 0.000000\r\n');
fprintf(fileID,'Rx Width (deg) = 30.00000\r\n');
fprintf(fileID,'Aperture Mode = 1\r\n');
fprintf(fileID,'Shading = 0\r\n');
fprintf(fileID,'-ve offsets = FALSE\r\n');
fprintf(fileID,'Tx Aperture (chan) = 1\r\n');
fprintf(fileID,'Rx Aperture (chan) = 1\r\n');
fprintf(fileID,'Allow Partial Apertures = FALSE\r\n');
fprintf(fileID,'First Element = 1\r\n');
fprintf(fileID,'Last Element = %d\r\n', number_of_elements);
fprintf(fileID,'Beam Step Value = 1.000000\r\n');
fprintf(fileID,'Beam Step = 4\r\n');
fprintf(fileID,'First Angle (deg) = 0.000000\r\n');
fprintf(fileID,'Last Angle (deg) = 45.000000\r\n');
fprintf(fileID,'Angle Pitch (deg) = 1.000000\r\n');
fprintf(fileID,'Scan Mode = 3\r\n');
fprintf(fileID,'Focus Mode = 0\r\n');
fprintf(fileID,'FocusX (mm) = 40.000000\r\n');
fprintf(fileID,'FocusY (mm) = 0.000000\r\n');
fprintf(fileID,'FocusZ (mm) = 0.000000\r\n\r\n');

%write out FMC sequence
fprintf(fileID,'[FMC sequence]\r\n');
fprintf(fileID,'First Tx = 1\r\n');
fprintf(fileID,'Num Tx = %d\r\n', number_of_elements);
fprintf(fileID,'Tx Step = 1\r\n');
fprintf(fileID,'First Rx = 1\r\n');
fprintf(fileID,'Num Rx = %d\r\n', number_of_elements);
fprintf(fileID,'Rx Step = 1\r\n\r\n');

%write out Transmit Recieve
fprintf(fileID,'[Transmit Receive]\r\n');
fprintf(fileID,'PRF Hz = %.5f\r\n', test_options.prf);
fprintf(fileID,'Tx Volts = %.5f\r\n', test_options.pulse_voltage);
fprintf(fileID,'TxFreq MHz = %.5f\r\n', (test_options.pulse_frequency));
fprintf(fileID,'NumCycles = %.5f\r\n', test_options.pulse_cycles);    
fprintf(fileID,'Polarity = FALSE\r\n');
fprintf(fileID,'%% active = %.5f\r\n', test_options.pulse_active);
fprintf(fileID,'PulseSpecMode = 0\r\n');
fprintf(fileID,'InitGain dB = %.5f\r\n', test_options.db_gain);
fprintf(fileID,'DACdelay us = %.5f\r\n', (test_options.gate_start*1e6));
fprintf(fileID,'DACslope dB/us = 0.000000\r\n');
fprintf(fileID,'SweptRange dB = 20.000000\r\n');
fprintf(fileID,'FixedGain = TRUE\r\n');
fprintf(fileID,'TGCslider dBmin = -6.000000\r\n');
fprintf(fileID,'TGCslider dBmax = 6.000000\r\n');
fprintf(fileID,'TGCslider dBstart = 0.000000\r\n');
fprintf(fileID,'TGCslider range(us) = 0.000000\r\n');
fprintf(fileID,'TGCslider spacing = 0\r\n');
fprintf(fileID,'TGCslider0 = 0\r\n');
fprintf(fileID,'TGCslider1 = 0\r\n');
fprintf(fileID,'TGCslider2 = 0\r\n');
fprintf(fileID,'TGCslider3 = 0\r\n');
fprintf(fileID,'TGCslider4 = 0\r\n');
fprintf(fileID,'TGCslider5 = 0\r\n');
fprintf(fileID,'TGCslider6 = 0\r\n');
fprintf(fileID,'TGCslider7 = 0\r\n\r\n');

%Display - not used by BRAIN but included for completeness
fprintf(fileID,'[Display]\r\n');
fprintf(fileID,'Range Start mm = 1.000000\r\n');
fprintf(fileID,'Range Finish mm = 11.000000\r\n');
fprintf(fileID,'Num Range Pix = 21\r\n');
fprintf(fileID,'Raster Start mm = -25.000000\r\n');
fprintf(fileID,'Raster Finish mm = 25.000000\r\n');
fprintf(fileID,'Num Raster Pix = 101\r\n');
fprintf(fileID,'Define Pixel size = TRUE\r\n');
fprintf(fileID,'Pixel size mm = 0.500000\r\n');
fprintf(fileID,'Velocity m/s = %.5f\r\n\r\n', ph_velocity);

%write out A-scan
fprintf(fileID,'[Ascan]\r\n');
fprintf(fileID,'Freq MHz = %.5f\r\n', (test_options.sample_freq/1e6));
fprintf(fileID,'Num Samples = %d\r\n', test_options.time_pts);
fprintf(fileID,'Sample Offset = %d\r\n\r\n', round(test_options.gate_start/(1/test_options.sample_freq)));

%filters - not currently supported
fprintf(fileID,'[Filter]\r\n');
fprintf(fileID,'AFE noise mode = FALSE\r\n');
fprintf(fileID,'AFE DCcoupling = FALSE\r\n');
fprintf(fileID,'AFE DigHPfilter = 8\r\n');
fprintf(fileID,'AFE LPfilter = 2\r\n');
fprintf(fileID,'HPfilter Enable = FALSE\r\n');
fprintf(fileID,'HPfilterStopMHz = 1.500000\r\n');
fprintf(fileID,'HPfilterPassMHz = 3.000000\r\n');
fprintf(fileID,'LPfilter Enable = FALSE\r\n');
fprintf(fileID,'LPfilterPassMHz = 6.000000\r\n');
fprintf(fileID,'LPfilterStopMHz = 7.000000\r\n');
fprintf(fileID,'Num Taps = 11\r\n\r\n');

%display options - not used by BRAIN
fprintf(fileID,'[Palette]\r\n');
fprintf(fileID,'Lower threshold = 11\r\n');
fprintf(fileID,'Upper threshold = 255\r\n');
fprintf(fileID,'Palette = 1\r\n');
fprintf(fileID,'Underrange colour = 16777215\r\n');
fprintf(fileID,'Overrange colour = 16777222\r\n');
fprintf(fileID,'B-scan Rectification = 1\r\n');
fprintf(fileID,'PostGain(dB) = 0.000000\r\n');
fprintf(fileID,'Gamma = 0\r\n\r\n');
fprintf(fileID,'[TOF Palette]\r\n');
fprintf(fileID,'Lower threshold = 11\r\n');
fprintf(fileID,'Upper threshold = 255\r\n');
fprintf(fileID,'Palette = 1\r\n');
fprintf(fileID,'Underrange colour = 16777215\r\n');
fprintf(fileID,'Overrange colour = 16777222\r\n\r\n');
fprintf(fileID,'[Display Orientation]\r\n');
fprintf(fileID,'Mirror = FALSE\r\n');
fprintf(fileID,'Rotate = 1\r\n');
fprintf(fileID,'Range Zoom = 0\r\n\r\n');

%gates - not sure these are used but setting for completeness
fprintf(fileID,'[Gate0]\r\n');
fprintf(fileID,'Enable = FALSE\r\n');
fprintf(fileID,'GateRange(us) = 0.000000\r\n');
fprintf(fileID,'GateWidth(us) = 10.000000\r\n');
fprintf(fileID,'Rectification = 2\r\n');
fprintf(fileID,'GateMode = 1\r\n');
fprintf(fileID,'Threshold = 30.000000\r\n');
fprintf(fileID,'Reject = 0.000000\r\n\r\n');
fprintf(fileID,'[Gate1]\r\n');
fprintf(fileID,'Enable = TRUE\r\n');
fprintf(fileID,'GateRange(us) = %.5f\r\n', (test_options.gate_start * 1e6));
fprintf(fileID,'GateWidth(us) = %.5f\r\n', (((test_options.gate_start)+(test_options.time_pts*(1/test_options.sample_freq)))*1e6));
fprintf(fileID,'Rectification = 2\r\n');
fprintf(fileID,'GateMode = 0\r\n');
fprintf(fileID,'Threshold = 0.000000\r\n');
fprintf(fileID,'Reject = 0.000000\r\n\r\n');

%Encoder - scanning not supported by BRAIN
fprintf(fileID,'[Encoder0]\r\n');
fprintf(fileID,'EncoderPitch(mm) = 0.129300\r\n');
fprintf(fileID,'StepAndDir = FALSE\r\n');
fprintf(fileID,'Reverse = TRUE\r\n');
fprintf(fileID,'Tracks = TRUE\r\n');
fprintf(fileID,'CscanPitch(mm) = 0.000000\r\n');
fprintf(fileID,'StartPosition(mm) = 0.000000\r\n\r\n');

%Aquire Mode
fprintf(fileID,'[AcquireMode]\r\n');
fprintf(fileID,'ArrayConfig = 1\r\n');
fprintf(fileID,'TxArrayNum = 0\r\n');
fprintf(fileID,'RxArrayNum = 0\r\n');

%Close the file
fclose(fileID);
