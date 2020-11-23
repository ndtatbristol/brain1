
__device__ float lanczos_interpolation2(const float* __restrict__ x, const float t, const int n,const float a, const float* lcz2, const int Nlcz)
{
    int i_min=(int)t - a + 1;
    int i_max=(int)t + a;
    float val=0.0F;
    if (i_min < 0 || i_max >= n-1)
    {
        return 0.0F;
    }
    else
    {
        for (int i=i_min; i<=i_max; i++)
        {
            float idx=(t-i+a)/(2*a)*(Nlcz-1);
            int idxI=(int)(idx);
            float idx0=lcz2[idxI]; float idx1=lcz2[idxI+1];
            float factor=(idx-idxI)*(idx1-idx0)+idx0;
            val+=x[i]*factor;
        }
    }
    return val;
}



__global__ void gpu_tfm_lanczos2_arb_2dly(float* real_result,float* imag_result,const int n,const int combs, const float* real_exp,const float* img_exp,const int* transmit,const int* receive,const float* lookup_time_tx,const float* lookup_time_rx,const float* time, const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float* lookup_amp_tx, const float* lookup_amp_rx, const float* tt_weight, const float aFactor){

	// get pixel's coordinates
#define NLanczos 1001
    __shared__ float lanczos_window2[NLanczos];
    float pi=asin(1.0F)*2.0F;
    //if (threadIdx.x == 0)
    //{
    //for (int pix = 0;pix < NLanczos; pix ++) 
    //   {
        for (int pix = threadIdx.x;pix < NLanczos; pix += blockDim.x) 
       {
            float t=pix*2.0*aFactor/(1.0*(NLanczos-1))-aFactor;
            if (abs(t) > aFactor){lanczos_window2[pix]=0.0;}
            else if (abs(t) < 1e-7){lanczos_window2[pix]=1.0;}
            else
            {
                float pit=pi*t;
                lanczos_window2[pix]=aFactor*sin(pit/aFactor)*sin(pit)/(pit*pit);
            }
       }
   //}
   __syncthreads();
   
   int NLanczos2=NLanczos;
   
   for (int pix = blockIdx.x * blockDim.x + threadIdx.x;pix < tot_pix; pix += blockDim.x * gridDim.x) 
   {
            //local variable
            float tot_real = 0, tot_imag = 0;
            float dt = time[1]-time[0];
            float invdt = 1.0F/dt;
            
            for(int ii = 0; ii < combs; ii++){
                //float real = 0;
                //float imag = 0;
                int tx = transmit[ii]-1;
                int rx = receive[ii]-1;
                int t_ind = (tx*grid_x*grid_y*grid_z)+pix;
                int r_ind = (rx*grid_x*grid_y*grid_z)+pix;

                float time_val = lookup_time_tx[t_ind] + lookup_time_rx[r_ind]; 
                float amp_corr = lookup_amp_tx[t_ind]*lookup_amp_rx[r_ind]*tt_weight[ii];
                float time_diff = time_val-time[0];
                if(time_diff<0){
                    }
                else if(time_val > time[n-1]){
                    }
                else
                    {
                    int scanline = ii*(n);
                    float lookup_index_float = time_diff * invdt;
                    // sum each val
                    tot_real += amp_corr*lanczos_interpolation2(&real_exp[scanline],lookup_index_float,n,aFactor,lanczos_window2,NLanczos2);
                    tot_imag += amp_corr*lanczos_interpolation2(&img_exp[scanline],lookup_index_float,n,aFactor,lanczos_window2,NLanczos2);
                    }   
			}
		
		// store the final value for the pixel
		real_result[pix] = tot_real;
        imag_result[pix] = tot_imag;
    }
}