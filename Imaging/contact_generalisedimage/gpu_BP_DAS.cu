/*
Update 29/08/2020

Generalized image:
g(xT,xR,z) = -1/(2pi c) SS z^2 / rT^1.5 / rR^1.5 u_t(xT',xR',t=(rT+rR)/c) dxT'dxR'

	where u_t is d/dt(u), or alternatively filtered by F(w)=jw

Input: u_t
*/


#define pi 3.141592f

// Forward imaging with FMC
__global__ void BP_DAS_FWD_FMC_complex(float2 * pOut, const float2 * pIn, int N, int Nx, int Nz, int Nt, float d, float dx, float dz, float dt, float c, float x0, float z0, float t0, float angFilt)
{
	int zn  = threadIdx.x + blockDim.x * blockIdx.x;
	int xRn = threadIdx.y + blockDim.y * blockIdx.y;
	int xTn = threadIdx.z + blockDim.z * blockIdx.z;
	
	float xT = xTn * dx + x0;
	float xR = xRn * dx + x0;
	float z	 = zn  * dz + z0;
	
	if( xTn<Nx && xRn<Nx && zn<Nz )
	{
		int idxOut 		= zn + xRn*Nz  + xTn*Nz*Nx;
		
		float Areal = 0;
        float Aimag = 0;

	
		for(int tx = 0; tx<N; tx++)
		{
			float Xtx = (float) tx*d - (N-1)*d/2;
			float Rtx = sqrtf( (xT-Xtx)*(xT-Xtx) + z*z );
			
			float invRtx2 = 1/(Rtx*Rtx);
			float sqrtRtx = sqrtf(Rtx);
			
			float BeamspreadTx = 1;
			if( Rtx>0 )			BeamspreadTx = z * invRtx2 * sqrtRtx ;
					
            if( (z/Rtx > __cosf(angFilt*pi/180)) || angFilt==0 )
			for(int rx = 0; rx<N; rx++)
			{
				float Xrx = (float) rx*d - (N-1)*d/2;
				float Rrx = sqrtf( (xR-Xrx)*(xR-Xrx) + z*z );
				
				float ToF  = (Rtx + Rrx)/c;
				int   ToFn = floorf((ToF-t0)/dt);
				float ToFi = ToFn*dt + t0;
				
                if( (z/Rrx > __cosf(angFilt*pi/180)) || angFilt==0 )
				if(ToFn>0 && ToFn<Nt-1)
				{
					
					int idxIn  		=  ToFn     + rx*Nt    + tx*Nt*N;
					int idxIn1 		= (ToFn+1)  + rx*Nt    + tx*Nt*N;

					if( idxIn>0 && idxIn<N*N*Nt-1)
					{
						
						float invRrx2 = 1/(Rrx*Rrx);
						float sqrtRrx 		= sqrtf(Rrx);
						
						float BeamspreadRx = 1;
						if( Rrx>0 )			BeamspreadRx = z * invRrx2 * sqrtRrx ;
						
						float coeff 	= BeamspreadTx * BeamspreadRx;

						Areal += (pIn[idxIn].x + (pIn[idxIn1].x-pIn[idxIn].x)/dt*(ToF - ToFi))*coeff;
                        Aimag += (pIn[idxIn].y + (pIn[idxIn1].y-pIn[idxIn].y)/dt*(ToF - ToFi))*coeff;
						
						
					}
				}
				
			}
		}
		pOut[idxOut].x = -Areal*d*d/2/pi/c;
        pOut[idxOut].y = -Aimag*d*d/2/pi/c;
		//__syncthreads();
	}
}

// Forward imaging with HMC
__global__ void BP_DAS_FWD_HMC_complex(float2 * pOut, const float2 * pIn, int N, int Nx, int Nz, int Nt, float d, float dx, float dz, float dt, float c, float x0, float z0, float t0, float angFilt)
{
	int zn  = threadIdx.x + blockDim.x * blockIdx.x;
	int xRn = threadIdx.y + blockDim.y * blockIdx.y;
	int xTn = threadIdx.z + blockDim.z * blockIdx.z;
	
	float xT = xTn * dx + x0;
	float xR = xRn * dx + x0;
	float z	 = zn  * dz + z0;
	
	if( xTn<Nx && xRn<Nx && zn<Nz )
	{
		int idxOut 		= zn + xRn*Nz  + xTn*Nz*Nx;
        int idxOut_sym	= zn + xTn*Nz  + xRn*Nz*Nx;
		
		float Areal = 0;
        float Aimag = 0;

	
		for(int tx = 0; tx<N; tx++)
		{
			float Xtx = (float) tx*d - (N-1)*d/2;
			float Rtx = sqrtf( (xT-Xtx)*(xT-Xtx) + z*z );
			
			float invRtx2 = 1/(Rtx*Rtx);
			float sqrtRtx = sqrtf(Rtx);
			
			float BeamspreadTx = 1;
			if( Rtx>0 )			BeamspreadTx = z * invRtx2 * sqrtRtx ;
					
            if( (z/Rtx > __cosf(angFilt*pi/180)) || angFilt==0 )
			for(int rx = tx; rx<N; rx++)
			{
				float Xrx = (float) rx*d - (N-1)*d/2;
				float Rrx = sqrtf( (xR-Xrx)*(xR-Xrx) + z*z );
				
				float ToF  = (Rtx + Rrx)/c;
				int   ToFn = floorf((ToF-t0)/dt);
				float ToFi = ToFn*dt + t0;


				float Rtx_sym = sqrt( (xT-Xrx)*(xT-Xrx) + z*z );
				float Rrx_sym = sqrt( (xR-Xtx)*(xR-Xtx) + z*z );
				
				float ToF_sym = (Rtx_sym + Rrx_sym)/c;
				int ToFn_sym = floorf((ToF_sym-t0)/dt);
                float ToFi_sym = ToFn_sym*dt + t0;

				
                if( (z/Rrx > __cosf(angFilt*pi/180)) || angFilt==0 )
				if(ToFn>0 && ToFn<Nt-1 && ToFn_sym>0 && ToFn_sym<Nt-1)
				{
					
                    int idxHMC = tx*N+rx - (tx+1)*tx/2;
					
					int idxIn  		=  ToFn     + idxHMC*Nt;
					int idxIn1 		= (ToFn+1)  + idxHMC*Nt;
					
					int idxIn_sym  		=  ToFn_sym     + idxHMC*Nt;
					int idxIn1_sym 		= (ToFn_sym+1)  + idxHMC*Nt;

					if( idxIn>0 && idxIn1<N*(N+1)/2*Nt && idxIn_sym>0 && idxIn1_sym<N*(N+1)/2*Nt)
					{
						
						float invRrx2 = 1/(Rrx*Rrx);
						
						float invRtx_sym2 = 1/(Rtx_sym*Rtx_sym);
						float invRrx_sym2 = 1/(Rrx_sym*Rrx_sym);
						
						float sqrtRrx 		= sqrtf(Rrx);
						float sqrtRtx_sym 	= sqrtf(Rtx_sym);
						float sqrtRrx_sym	= sqrtf(Rrx_sym);
						
						float BeamspreadRx = 1;
						if( Rrx>0 )			BeamspreadRx = z * invRrx2 * sqrtRrx ;
						
						float BeamspreadRx_sym = 1;
						if( Rrx_sym>0 )		BeamspreadRx_sym = z * invRrx_sym2 * sqrtRrx_sym ;
						
						float BeamspreadTx_sym = 1;
						if( Rtx_sym>0 )		BeamspreadTx_sym = z * invRtx_sym2 * sqrtRtx_sym ;
						
						float coeff 	= BeamspreadTx * BeamspreadRx;
						float coeff_sym = BeamspreadTx_sym * BeamspreadRx_sym;
	
                        Areal += (pIn[idxIn].x + (pIn[idxIn1].x-pIn[idxIn].x)/dt*(ToF - ToFi))*coeff;
                        Aimag += (pIn[idxIn].y + (pIn[idxIn1].y-pIn[idxIn].y)/dt*(ToF - ToFi))*coeff;
										
						if(rx!=tx)
                        {
                         	Areal += (pIn[idxIn_sym].x + (pIn[idxIn1_sym].x-pIn[idxIn_sym].x)/dt*(ToF_sym - ToFi_sym))*coeff_sym;
                            Aimag += (pIn[idxIn_sym].y + (pIn[idxIn1_sym].y-pIn[idxIn_sym].y)/dt*(ToF_sym - ToFi_sym))*coeff_sym;
                        }

						
					}
				}
				
			}
		}
		pOut[idxOut].x = -Areal*d*d/2/pi/c;
        pOut[idxOut].y = -Aimag*d*d/2/pi/c;
        if(xTn!=xRn)
        { 
            pOut[idxOut_sym].x = -Areal*d*d/2/pi/c;
            pOut[idxOut_sym].y = -Aimag*d*d/2/pi/c;
        }
		//__syncthreads();
	}
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
Update 12/03/2020

Recovered data:
u(xT,xR,t) = 1/(2pi c) d/dt SS 1 / rT^0.5 / rR^0.5 g(xT',xR',z=zeta) dxT'dxR'

	where zeta = sqrt( (ct+(DxT+DxR))*(ct-(DxT+DxR))*(ct+(DxT-DxR))*(ct-(DxT-DxR)) ) / (2ct)

Output: 1/(2pi c) SS 1 / rT^0.5 / rR^0.5 g(xT',xR',z=zeta) dxT'dxR'
Calculate the derivative outside.
*/

// Inverse imaging with FMC
__global__ void BP_DAS_INV_FMC(float * pOut, const float * pIn, int N, int Nx, int Nz, int Nt, float d, float dx, float dz, float dt, float c, float x0, float z0, float t0, float Twidth, float posX, float posZ)
{
	int tn = threadIdx.x + blockDim.x * blockIdx.x;
	int rx = threadIdx.y + blockDim.y * blockIdx.y;
	int tx = threadIdx.z + blockDim.z * blockIdx.z;
	
	float Xtx = (float) tx*d - (N-1)*d/2;
	float Xrx = (float) rx*d - (N-1)*d/2;
	float t   = (float) tn*dt + t0;
	
	float tc = (sqrtf( (Xtx-posX)*(Xtx-posX) + posZ*posZ ) + sqrtf( (Xrx-posX)*(Xrx-posX) + posZ*posZ))/c ;
	
	if( abs( t-tc) < Twidth/2 )
	if( tx<N && rx<N && tn<Nt )
	{
		int idxOut 		= tn + rx*Nt  + tx*Nt*N;

		float Amplitude = 0;
	
		for(int xTn = 0; xTn<Nx; xTn++)
		{
			float xT = xTn * dx + x0;
			
			for(int xRn = 0; xRn<Nx; xRn++)
			{
				float xR = xRn * dx + x0;
				
				float ch1 = (c*c*t*t + (xR-Xrx)*(xR-Xrx) - (xT-Xtx)*(xT-Xtx))/(2*c*t);
				float ch2 = ch1*ch1 - (xR-Xrx)*(xR-Xrx);
				
				
				if( ch2>=0 )
				{
					float z = sqrtf( ch2 );
					
					int zn = floorf((z-z0)/dz);
					float zi = zn*dz + z0;
					
					if(zn>0 && zn<Nz-1)
					{

						float Rtx = sqrtf( (xT-Xtx)*(xT-Xtx) + z*z );
						float Rrx = sqrtf( (xR-Xrx)*(xR-Xrx) + z*z );
						
						float coeff = 1 / sqrtf(Rtx*Rrx);
						
						
						int idxIn  		= zn    + xRn*Nz    + xTn*Nz*Nx;
						int idxIn1 		= zn+1  + xRn*Nz    + xTn*Nz*Nx;
						
						if( idxIn>0 && idxIn<Nx*Nx*Nz-1)
						{
							
							Amplitude += (pIn[idxIn] + (pIn[idxIn1]-pIn[idxIn])/dz*(z - zi)) *coeff;
							
						}
					}
				}
				
			}
		}
		pOut[idxOut] = Amplitude*dx*dx/2/pi/c;
		//__syncthreads();
	}
}


// Inverse imaging with HMC
__global__ void BP_DAS_INV_HMC(float * pOut, const float * pIn, int N, int Nx, int Nz, int Nt, float d, float dx, float dz, float dt, float c, float x0, float z0, float t0, float Twidth, float posX, float posZ)
{
	int tn = threadIdx.x + blockDim.x * blockIdx.x;
	int rx = threadIdx.y + blockDim.y * blockIdx.y;
	int tx = threadIdx.z + blockDim.z * blockIdx.z;
	
	float Xtx = (float) tx*d - (N-1)*d/2;
	float Xrx = (float) rx*d - (N-1)*d/2;
	float t   = (float) tn*dt + t0;
	
	float tc = (sqrtf( (Xtx-posX)*(Xtx-posX) + posZ*posZ ) + sqrtf( (Xrx-posX)*(Xrx-posX) + posZ*posZ))/c ;
	
	if( abs( t-tc) < Twidth/2 )
	if( tx<N && tx<=rx && tn<Nt )
	{
		//int idxOut 		= tn + rx*Nt  + tx*Nt*N;
		//int idxOut = tn + ((tx+1)*(tx+2)/2-(tx-rx)-1) * Nt;
		int idxOut = tn + (tx*N+rx - (tx+1)*tx/2) * Nt;
		
       // int idxOut = tn + (tx*N+rx) * Nt;

		float Amplitude = 0;
	
		for(int xTn = 0; xTn<Nx; xTn++)
		{
			float xT = xTn * dx + x0;
			
			for(int xRn = xTn; xRn<Nx; xRn++)
			{
				float xR = xRn * dx + x0;
				
				float ch1 = (c*c*t*t + (xR-Xrx)*(xR-Xrx) - (xT-Xtx)*(xT-Xtx))/(2*c*t);
				float ch2 = ch1*ch1 - (xR-Xrx)*(xR-Xrx);
				
				float ch1_sym = (c*c*t*t + (xT-Xrx)*(xT-Xrx) - (xR-Xtx)*(xR-Xtx))/(2*c*t);
				float ch2_sym = ch1_sym*ch1_sym - (xT-Xrx)*(xT-Xrx);
				
				
				if( ch2>=0 && ch2_sym>=0)
				{
					float z = sqrtf( ch2 );
					int zn = floorf((z-z0)/dz);
					
					float z_sym = sqrtf( ch2_sym );
					int zn_sym = floorf((z_sym-z0)/dz);
					
					if(zn>0 && zn<Nz-1 && zn_sym>0 && zn_sym<Nz-1)
					{

						float Rtx = sqrtf( (xT-Xtx)*(xT-Xtx) + z*z );
						float Rrx = sqrtf( (xR-Xrx)*(xR-Xrx) + z*z );
						
						float Rtx_sym = sqrtf( (xR-Xtx)*(xR-Xtx) + z_sym*z_sym );
						float Rrx_sym = sqrtf( (xT-Xrx)*(xT-Xrx) + z_sym*z_sym );
						
						float coeff 	= 1 / sqrtf(Rtx*Rrx);
						float coeff_sym = 1 / sqrtf(Rtx_sym*Rrx_sym);
						
						
						int idxIn  		= zn    + xRn*Nz    + xTn*Nz*Nx;
						int idxIn1 		= zn+1  + xRn*Nz    + xTn*Nz*Nx;
						
						int idxIn_sym  		= zn_sym    + xRn*Nz    + xTn*Nz*Nx;
						int idxIn1_sym 		= zn_sym+1  + xRn*Nz    + xTn*Nz*Nx;
						
						if( idxIn>0 && idxIn<Nx*Nx*Nz-2 && idxIn_sym>0 && idxIn_sym<Nx*Nx*Nz-2)
						{
							
							float vi     = (pIn[idxIn] + (pIn[idxIn1]-pIn[idxIn])/dz*(z - zn*dz))*coeff;
							float vi_sym = (pIn[idxIn_sym] + (pIn[idxIn1_sym]-pIn[idxIn_sym])/dz*(z_sym - zn_sym*dz))*coeff_sym;
							
							
							if(xTn!=xRn) 	Amplitude += vi + vi_sym;
							else			Amplitude += vi;

						}
					}
				}
				
			}
		}
		pOut[idxOut] = Amplitude*dx*dx/2/pi/c;

	}
}

