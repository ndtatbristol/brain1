/*===============================================================================
$Header: /PNL/mptcp/source/mptcp.h,v 1.1 2005/05/19 12:41:26 RobertCain Exp $
Copyright © Peak NDT Ltd. 2000,2005 - All Rights Reserved.

File      : mptcp.h
Function  : example c header file for mptcp.dll
------------------------------------------------------------------------------
$Log: mptcp.h,v $
Revision 1.1  2005/05/19 12:41:26  RobertCain
Users need example header file esp for func names.

===============================================================================*/


#define			IBNOTIFY	"_ibnotify@16"
#define			IBDEV		"_ibdev@24"
#define			IBWRT		"_ibwrt@12"
#define			IBRD		"_ibrd@12"
#define			IBRSP		"_ibrsp@8"
#define			IBSTA		"_ibsta@0"
#define			IBCNT		"_ibcnt@0"

#define			ibsta		pibsta()
#define			ibcnt		pibcnt()

/* ----------------------------------------------------------------------------
 	PROTOTYPES FOR DLL
   ----------------------------------------------------------------------------*/

#ifdef __cplusplus
	extern "C" {
#endif

typedef			int ( __stdcall *GpibNotifyCallback_t   )( int, int, int, long, void * );
typedef			int ( __stdcall *PFNIBNOTIFY            )( int, int, GpibNotifyCallback_t, void * );
typedef			int	( __stdcall *PFNIBDEV               )(  int, int, int, int, int, int );
typedef			int ( __stdcall *PFNIBWRT               )( int, void *, long );
typedef			int ( __stdcall *PFNIBRD                )( int ud, void *Buffer, long count );
typedef			void( __stdcall *PFNSENDIFC             )( int );
typedef			int ( __stdcall *PFNIBSTA               )( void );
typedef			int ( __stdcall *PFNIBCNT               )( void );
typedef			int ( __stdcall *PFNIBRSP               )( void );

#ifdef __cplusplus
	}
#endif
/*
PFNIBNOTIFY		ibnotify;
PFNIBDEV		ibdev;
PFNIBWRT		ibwrt;
PFNIBRD			ibrd;
PFNIBRSP		ibrsp;
PFNSENDIFC		SendIFC;
PFNIBSTA		pibsta;
PFNIBCNT		pibcnt;
*/
