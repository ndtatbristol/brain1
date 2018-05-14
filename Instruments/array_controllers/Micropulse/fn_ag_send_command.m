%COMMAND
%   [result, err_msg] = fn_ag_send_command(cmd_str, timeout [,echo_on])
%SUMMARY
%   Primary communication function for communicating with Micropulse,
%   implemented via mex file fn_ag_send_command.c, which must be compiled on
%   compatible compiler. All commands are sent by this function which also
%   deals with returned messages. However, common command sequences are
%   implemented in normal m-file "wrapper" functions and fn_ag_send_command
%   should not normally be used directly for most work.
%INPUTS
%   cmd_str - text string of command characters. These are the commands in
%   the Micropulse manual with the addition of a few extra ones to deal
%   with establishing the initial connection and clearing the read buffer.
%   timeout - the timeout (in seconds) before function returns if no data
%   has been returned by Micropulse. If zero, the function sends the 
%   command string to Micropulse and returns immediately - this should be
%   used for commands with no return data associated with them. If timeout
%   of zero is used with a command that does return data then the data
%   will still be read by the PC but it cannot then be accessed from
%   Matlab (an 'Unexpected data!' message will be shown if echo_on = 1).
%   echo_on [0] - if one, communications are echoed on the screen (slower, but
%   useful for debugging).
%OUTPUTS
%   result - the result returned by Micropulse in the form of a single
%   vector of unsigned integer bytes. Empty vector if nothing returned.
%   err_msg - error message.
%NOTES
%   The following extra commands (not in Micropulse Manual) are used:
%   cmd_str = 'CONNECT xxx.xxx.xxx.xxx yyyy' to establish the connection
%   where xxx.xxx.xxx.xxx is the IP address and yyyy is the port number.
%   This function can be called repeatedly - if Micropulse is already
%   connected then it is ignored.
%   cmd_str = 'READ' to force a read of the read buffer. Possibly useful
%   after a crash if there is still data to be read.
%KNOWN BUGS
%   The "connect" command must be used with echo_on = 1. If there are
%   connection problems they are usually within either mptcp DLL.
%   Therefore first try exiting Matlab and restarting it (i.e. no need to
%   turn the instrument off and on again) as this usually is all that is
%   needed as it forces the DLLs to unload and be reloaded.