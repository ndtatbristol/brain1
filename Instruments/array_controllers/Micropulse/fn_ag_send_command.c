/*=================================================================
 *
 * fn_ag_send_command.c
 *		Source code to send commands and receive data from
 *		Peak NDT Array Controller
 *
 * The calling syntax for sending comamands and receiving data is:
 *
 *		[result, error_code] =
 *			fn_ag_send_command(cmd_str, timeout [, echo_on]);
 *
 * Written by Paul Wilcox (January 2007)
 *
 *=================================================================*/
#include <math.h>
#include <windows.h>
#include <time.h>
#include <string.h>
#include <stdio.h>
#include <mex.h>
#include "matrix.h"
#include "mptcp.h"

//Hardware and DLL defs
#define DLL_NAME		"mptcp.dll"
#define UD              0

//Extreme test params - used to set max buffer size
#define MAX_CH          64
#define MAX_PTS_PER_CH  10000
#define MAX_BYTES_PER_PT    2
#define MAX_HEADER_LENGTH   100
#define MAIN_BUFFER_LENGTH  MAX_PTS_PER_CH * MAX_CH * MAX_CH * MAX_BYTES_PER_PT + MAX_HEADER_LENGTH

//Various command strings requiring special actions
#define READ_ONLY_STR   "READ"
#define CONNECT_STR     "CONNECT"

//Various timeouts
#define WRITE_TIMEOUT   5
#define DELAY_BETWEEN_WRITES    0.5
#define CONNECT_DELAY   1

//IEEE ibnotify and Peak NDT header mask codes
#define NOTIFY_MASK     0x0800
#define ASCAN_HDR_MASK      0x1A
 
//Mex file arguments
#define	CMD_STR			prhs[0]
#define	TIMEOUT			prhs[1]
#define	DEBUG			prhs[2]

#define	RESULT			plhs[0]
#define	ERROR_MSG		plhs[1]

//Static global variables
static int              initialised = 0;
static mxArray          *buffer = NULL;
static bool             connected = 0;

//Other global variables
int                     bytesRead = 0;
bool                    dataReady = 0;
bool                    resetBuffer = 1;
bool                    debug = 0;
bool                    dataExpected = 0;
bool                    keepReading = FALSE;
int                     bufferPos = 0;

HINSTANCE               hLib = NULL;
PFNIBNOTIFY             ibnotify = NULL;
PFNIBDEV                ibdev = NULL;
PFNIBWRT                ibwrt = NULL;
PFNIBRD                 ibrd = NULL;
PFNIBRSP                ibrsp = NULL;
PFNSENDIFC              SendIFC = NULL;
PFNIBSTA                pibsta = NULL;
PFNIBCNT                pibcnt = NULL;
GpibNotifyCallback_t    pCallback = NULL;

//function prototypes
void pause(double);
void readData(void);
int sendCommand(char *, double);
static void exitFunction(void);
static int __stdcall callback(int, int, int, long, void *);
int specifyCallbackFunction(void);
int loadLibrary(void);
int connectInstrument(int, int);
///////////////////////////////////////////////////////////////////////////

int sendCommand(char *cmd_str, double timeout) {
    int charSentCount, success, ip_address, port_no;
    double timestart, endtime;
    //load library if it isn't already
    if (!hLib) {
        if (!loadLibrary()) {
            return 0;
        }
    }
    
    //check connected to instrument and connect if not
    if (!strncmp(cmd_str, CONNECT_STR, strlen(CONNECT_STR))) {
        if (!connected) {
            sscanf(cmd_str, "%*s %d %d %d", &ip_address, &port_no);
            if (!connectInstrument(ip_address, port_no)) {
                connected = 0;
                return 0;
            } else {
                connected = 1;
                return 1;
            }
        } else {
                if (debug) mexPrintf("  Already connected\n");
                return 1;
        }
    }
    if (!connected) {
        if (debug) mexPrintf("  Must connect to instrument first !\n");
        return 0;
    };
    
    //prepare global variables for impending read
    resetBuffer = 1;
    bufferPos = 0;
    dataReady = 0;
    bytesRead = 0;
    timestart = (double)clock() / (double)CLK_TCK;
    endtime = timestart + timeout;
    if (timeout>0.0) {
        dataExpected = 1;
    } else {
        dataExpected = 0;
    };
    if (!strcmp(cmd_str, READ_ONLY_STR)) {
        //execute initial read if cmd_str = READ_ONLY_STR
        keepReading = TRUE;
        readData();
/*        if (debug) mexPrintf("  ibrd(%i, <buffer>, <%i) ... ", UD, MAIN_BUFFER_LENGTH);
        bytesRead = ibrd(UD, (char*)mxGetData(buffer), MAIN_BUFFER_LENGTH);
        if (debug) mexPrintf("%i\n", bytesRead);
        return bytesRead;*/
    } else {
        //actually send the command
        keepReading = FALSE;
        if (debug) mexPrintf("  ibwrt(%i, \"%s\", %i) ... ", UD, cmd_str, strlen(cmd_str));
        charSentCount = ibwrt(UD, cmd_str, strlen(cmd_str));
        if (debug) mexPrintf("%i bytes sent\n", charSentCount);
        if (charSentCount != strlen(cmd_str)) {
            return -2;//code for fail to send
        };
    };
    
    //if no return data is expected return
    if (timeout <= 0.0) {
        return -1;//-1 signifies successful write with no data read back
    };
    //wait for return data if some is expected (timeout > 0)
    while ((double)clock() / (double)CLK_TCK < endtime && !dataReady) {};
    if (!bytesRead) {
        return 0; //signfies successful write but failed to read any data back although some was expected
    };
    return bytesRead;//>1 signifies successful write with data read back
}

static void exitFunction(void) {
    mexPrintf("Exit function\n");
    mxFree(buffer);
}

void mexFunction( int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[] )
{
    char *cmd_str;
    char *output;
    double timeout,timestart, endtime;
    int buflen, ii;
    bool status, commandSent;
    // Check for proper number and form of arguments
    if (nrhs > 3) {
        mexErrMsgTxt("  Two or three input arguments required.");
    } else if (nlhs > 2) {
        mexErrMsgTxt("  Too many output arguments.");
    }
    if (mxIsChar(CMD_STR) != 1) {
        mexErrMsgTxt("  cmd_str must be a string.");
    }
    // convert inputs to C variables
    buflen = (mxGetM(CMD_STR) * mxGetN(CMD_STR)) + 1;
    cmd_str = mxCalloc(buflen, sizeof(char));
    status = mxGetString(CMD_STR, cmd_str, buflen);
    if (status != 0) {
        mexWarnMsgTxt("  Not enough space. String is truncated.");
    }
    timeout = mxGetScalar(TIMEOUT);
    if (nrhs == 3) {
        debug = mxGetScalar(DEBUG);
    } else {
        debug = 0;
    };
    
    //initialise buffer
    if (!initialised) {
        if (debug) mexPrintf("  Initialising result matrix\n");
        buffer = mxCreateNumericMatrix(1, MAIN_BUFFER_LENGTH, mxUINT8_CLASS, mxREAL);
        mexMakeArrayPersistent(buffer);
        mexAtExit(exitFunction);
        initialised = 1;
    }
    
    //try and send the command
    commandSent = 0;
    timestart = (double)clock() / (double)CLK_TCK;
    endtime = timestart + WRITE_TIMEOUT;
    while ((double)clock() / (double)CLK_TCK < endtime && !commandSent) {
        bytesRead = sendCommand(cmd_str, timeout);
        if (bytesRead != -2) {
            commandSent = 1;
        } else {
            pause(DELAY_BETWEEN_WRITES);
        }
    };
    
    if (bytesRead > 0) {
        //return contents of buffer and no error
        mxSetN(buffer, bytesRead);
        mxSetM(buffer, 1);
        RESULT = mxDuplicateArray(buffer);
        ERROR_MSG = mxCreateString("No errors");
    };
    if (bytesRead == -1) {
        //return empty matrix and no error
        RESULT = mxCreateNumericMatrix(0, 0, mxUINT8_CLASS, mxREAL);
        ERROR_MSG = mxCreateString("No errors");
    };
    if (bytesRead == -2) {
        //return empty matrix and no error
        RESULT = mxCreateNumericMatrix(0, 0, mxUINT8_CLASS, mxREAL);
        ERROR_MSG = mxCreateString("Error - timed out on write");
    };
    if (!bytesRead) {
        //return empty matrix and error
        RESULT = mxCreateNumericMatrix(0, 0, mxUINT8_CLASS, mxREAL);
        ERROR_MSG = mxCreateString("Error no response");
    }
}

static int __stdcall callback(int i1, int i2, int i3, long i4, void * i5) {
    int br;
    if (dataExpected) {
    readData();
    } else {
        if (debug) mexPrintf("  Unexpected data! ibrd(%i, <buffer>, %i) ... ", UD, MAIN_BUFFER_LENGTH);
        br = ibrd(UD, (char*)mxGetData(buffer) + bufferPos, MAIN_BUFFER_LENGTH);
        if (debug) mexPrintf("%i bytes read\n", br);
    }
    return NOTIFY_MASK;
}

void readData(void) {
    //this function either fills the buffer with all available data
    //or appends the current data to the buffer if its an A-scan and does not
    //set the dataReady flag
    int br, ii;
    //read whats in the TCPIP buffer to buffer variable starting at bufferPos position
    if (debug) mexPrintf("  ibrd(%i, <buffer>, %i) ... ", UD, MAIN_BUFFER_LENGTH - bufferPos);
    br = ibrd(UD, (char*)mxGetData(buffer) + bufferPos, MAIN_BUFFER_LENGTH - bufferPos);
    
    bytesRead = bytesRead + br;
    //special case for actual ASCAN or keepReading data - read is performed and data is 
    //appended to buffer but dataReady flag is not set
    if (((char*)mxGetData(buffer))[bufferPos] == ASCAN_HDR_MASK || keepReading) {
        if (resetBuffer) { //global variable resetBuffer is set to one when ibwrt is called
            resetBuffer = 0; //so subsequent A-scans are appended
            bufferPos = 0; //and first one starts at pos zero
        }
        if (debug) mexPrintf("A-scan ... %i bytes read\n", br);
        bufferPos = bufferPos + br;
        dataReady = 0; //data is NOT ready until end message is received
    }
    else {
        if (debug) mexPrintf("non A-scan data ... %i bytes read\n", br);
        bufferPos = bufferPos + br;
        dataReady = 1;
    };
    return;
};

int loadLibrary(void) {
    if (debug) mexPrintf("  Loading library ... ");
    hLib = LoadLibrary(DLL_NAME);
    if (hLib) {
        if (debug) mexPrintf("success\n");
        ibwrt = (PFNIBWRT)GetProcAddress(hLib, "ibwrt");
        ibrd = (PFNIBRD)GetProcAddress(hLib, "ibrd");
        ibdev = (PFNIBDEV)GetProcAddress(hLib, "ibdev");
        ibnotify = (PFNIBNOTIFY)GetProcAddress(hLib, "ibnotify");
        pibcnt = (PFNIBCNT)GetProcAddress(hLib, "ibcnt");
        pibsta = (PFNIBSTA)GetProcAddress(hLib, "ibsta");
        SendIFC = (PFNSENDIFC)GetProcAddress(hLib, "SendIFC");
    }
    else {
        if (debug) mexPrintf("failed\n");
        return 0;
    }
    return 1;
}

int connectInstrument(int ip_address, int port_no) {
    if (debug) mexPrintf("  ibdev(%i, %i, %i, 0, 0, 0) ... ", UD, ip_address, port_no);
    if (ibdev(UD, ip_address, port_no, 0, 0, 0) != 0) {
        if (debug) mexPrintf("failed\n");
        return 0;
    } else {
        if (debug) mexPrintf("success\n");
        pause(CONNECT_DELAY);
    };
    
    if (!specifyCallbackFunction()) {
        pause(CONNECT_DELAY);
        return 0;
    };
    return 1;
}

int specifyCallbackFunction(void) {
    int success;
    int pdw = 17;
    pCallback = (GpibNotifyCallback_t)callback;
    if (debug) mexPrintf("  ibnotify(%i, %i, %i, %i) ... ", UD, NOTIFY_MASK, (int)pCallback, pdw),
    success = ibnotify(UD, NOTIFY_MASK, pCallback, &pdw);
    if (success == 0) {
        if (debug) mexPrintf("success\n");
        return 1;
    }
    else {
        return 0;
        if (debug) mexPrintf("failed\n");
    }
}

void pause(double delay) {
    double timestart, endtime;
    timestart = (double)clock() / (double)CLK_TCK;
    endtime = timestart + delay;
    while ((double)clock() / (double)CLK_TCK < endtime) {};
}
