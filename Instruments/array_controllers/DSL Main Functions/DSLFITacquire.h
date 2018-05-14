#include "extcode.h"
#pragma pack(push)
#pragma pack(1)

#ifdef __cplusplus
extern "C" {
#endif
typedef uint16_t  FileOperation;
#define FileOperation_SaveConfiguration 0
#define FileOperation_LoadConfiguration 1
#define FileOperation_SaveFWCData 2
#define FileOperation_LoadFWCData 3
#define FileOperation_SaveFRDData 4
#define FileOperation_LoadFRDData 5

/*!
 * Splash screen launcher for DSLFITfocusStream.
 */
int32_t __stdcall LaunchDSLFITscan(char VIname[], char CallerID[], 
	char SystemConfigFile[], char ConfigFile[]);
/*!
 * VI to send an external request to set to a specified state and to handle 
 * the acknowledgement.
 * Used as a primitive within the DLL for handling external requests to force 
 * into a particular state.
 */
int32_t __stdcall SetState(char State[], int32_t timeoutms, 
	char ResponseMessage[], int32_t ResponseMessageLength);
/*!
 * VI to request the system to load or save the specified file, using the 
 * specified operation, and reports back with the filename if found and empty 
 * path constant if not.
 * The response message echoes with the state request sent.
 * The return value is set to 0 if the file was found, -1 iIf not found then 
 * the response file 
 */
int32_t __stdcall LoadSaveFile(FileOperation FileOperation, char FilePath[], 
	int32_t timeoutms, char ResponseFilePath[], int32_t ResponseFilePathLength, 
	char ResponseMessage[], int32_t ResponseMessageLength);
/*!
 * VI to send an external request to set or get the current value of the Gain 
 * & TGC/DAC and to handle the acknowledgement.
 * Used as a primitive within the DLL for handling external requests to force 
 * into a particular state.
 */
int32_t __stdcall SetGetGainTGC(int16_t Set, int32_t timeoutms, 
	double *MasterGainDB, int16_t *FixedGain, double *DACDelayus, 
	double *DACSlopeDBus, double *MaxGainDB, int8_t TGC[], int32_t TGClength);
/*!
 * VI to send an external request to get a specified Ascan.
 * Used as a primitive within the DLL for handling external requests to force 
 * into a particular state.
 */
int32_t __stdcall GetAscan(int32_t Frame, int32_t Tx, int32_t Rx, 
	int32_t Sample, int32_t timeoutms, int16_t Ascan[], int32_t AscanLength);
/*!
 * VI to send an external request to get the data size parameters.
 * Used as a primitive within the DLL for handling external requests to force 
 * into a particular state.
 */
int32_t __stdcall GetU64dataParas(int32_t timeoutms, int32_t *NumFrames, 
	int32_t *NumTx, int32_t *NumRx, int32_t *NumSamp);
/*!
 * VI to send an external request to get the raw U64format data reference 
 * along with its formatting parameters.
 * Used as a primitive within the DLL for handling external requests to force 
 * into a particular state.
 */
int32_t __stdcall GetU64dataStreamSegment(int32_t Frame, int32_t timeoutms, 
	int32_t U64startIndex, int32_t SizeU64segment, uint64_t U64stream[], 
	int32_t *NumTx, int32_t *NumRx, int32_t *NumSamp);
/*!
 * VI to send an external request to get the index value for a specific sample 
 * in the U64dataStream and the separation between it and the consecutive 
 * sample.
 * Used to build up the unpacking lookup table needed to extract the FMC & FRD 
 * data and is able to handle changes in the unpacking format.
 */
int32_t __stdcall GetU64streamIndexAndStep(int32_t Frame, int32_t Tx, 
	int32_t Rx, int32_t Sample, int32_t timeoutms, int32_t *U64index, 
	int32_t *U64indexSampleStep);
/*!
 * VI to send an external request to set or get the current value of the 
 * Transmit excitation and to handle the acknowledgement.
 * Used as a primitive within the DLL for handling external requests to force 
 * into a particular state.
 */
int32_t __stdcall SetGetTxExcitation(int16_t Set, int32_t timeoutms, 
	double *PRFHz, double *TxLevelV, uint16_t *PulseSpec, double *TxFreqHz, 
	double *Cycles, int16_t *Polarity, double *Active, uint16_t Train[], 
	int32_t TrainLength);
/*!
 * VI to send an external request to get the parameters for a specified Image 
 * and pixel using the Image parameter definitions in the LabVIEW Vision 
 * Manual.
 * Used as a primitive within the DLL for handling external requests.
 */
int32_t __stdcall GetImageData(int32_t Frame, int32_t Tx, int32_t Rx, 
	int32_t Sample, int32_t timeoutms, uint64_t *PixelPointerOut, 
	int32_t *PixelSizeBytes, int32_t *LineWidthPixels, int32_t *TransferMaxSize, 
	int32_t *ImageBorderSize, int32_t *XResolution, int32_t *Yresolution);
/*!
 * VI to send an external request to set or get a specified double preceision 
 * parameter and to handle the acknowledgement.
 * Used as a primitive within the DLL for handling external requests.
 */
int32_t __stdcall SetGetParaDouble(int16_t Set, int32_t timeoutms, 
	int32_t *Parameter, double Value[], int32_t ValueLength);
/*!
 * VI to send an external request to set or get a specified integer parameter 
 * and to handle the acknowledgement.
 * Used as a primitive within the DLL for handling external requests.
 */
int32_t __stdcall SetGetParaInt(int16_t Set, int32_t timeoutms, 
	int32_t *Parameter, int32_t Value[], int32_t ValueLength);
/*!
 * VI to send an external request to set or get a specified string parameter 
 * and to handle the acknowledgement.
 * Used as a primitive within the DLL for handling external requests.
 */
int32_t __stdcall SetGetParaString(int16_t Set, int32_t timeoutms, 
	int32_t *Parameter, char Value[], int32_t ValueLength);

/*
 * Call to get Ascans
 */
int32_t __stdcall GetCustomAscans(int32_t NumAscans, int32_t *Frames, 
	int32_t *Tx, int32_t *Rx, int32_t *Samples, int32_t timeoutms,
	int16_t *Ascans, int32_t AscansLength);

MgErr __cdecl LVDLLStatus(char *errStr, int errStrLen, void *module);

#ifdef __cplusplus
} // extern "C"
#endif

#pragma pack(pop)

