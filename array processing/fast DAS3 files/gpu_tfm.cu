#include<thrust/host_vector.h>
#include<thrust/device_vector.h>

__global__ void tfm_near_norm(float* real_result,float* imag_result,const int n,const int combs, const float* real_exp,const float* img_exp,const int* transmit,const int* receive,const int* lookup_ind, const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float* lookup_amp, const float* tt_weight){

	// get pixel's coordinates
    int pix = blockIdx.x*blockDim.x+threadIdx.x;
     
    if (pix<tot_pix){
            //local variable
            float tot_real = 0, tot_imag = 0;

            for(int ii = 0; ii < combs; ii++){
                float real = 0;
                float imag = 0;
                int tx = transmit[ii]-1;
                int rx = receive[ii]-1;
                int t_ind = (tx*grid_x*grid_y*grid_z)+pix;
                int r_ind = (rx*grid_x*grid_y*grid_z)+pix;

                int index = lookup_ind[t_ind] + lookup_ind[r_ind] -1; 
                float amp_corr = lookup_amp[t_ind]*lookup_amp[r_ind]*tt_weight[ii];
                if(index<0){
                    }
                else if(index>n){
                    }
                else
                    {
                    int set_val = ii*(n)+index;
                    real = real_exp[set_val];
                    real = real*amp_corr;
                    imag = img_exp[set_val];
                    imag = imag*amp_corr;
                    }
                // sum each val
                tot_real += real;
                tot_imag += imag;
					
            }
		
            // store the final value for the pixel
            //result[pix] = sqrt(tot_real*tot_real + tot_imag*tot_imag);
            real_result[pix] = tot_real;
            imag_result[pix] = tot_imag;
    }
}

__global__ void tfm_near_2dly(float* real_result,float* imag_result,const int n,const int combs, const float* real_exp,const float* img_exp,const int* transmit,const int* receive,const int* lookup_ind_tx, const int* lookup_ind_rx,const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float* lookup_amp_tx,const float* lookup_amp_rx,const float* tt_weight){

	// get pixel's coordinates
    int pix = blockIdx.x*blockDim.x+threadIdx.x;
     
    if (pix<tot_pix){
            //local variable
            float tot_real = 0, tot_imag = 0;

            for(int ii = 0; ii < combs; ii++){
                float real = 0;
                float imag = 0;
                int tx = transmit[ii]-1;
                int rx = receive[ii]-1;
                int t_ind = (tx*grid_x*grid_y*grid_z)+pix;
                int r_ind = (rx*grid_x*grid_y*grid_z)+pix;

                int index = lookup_ind_tx[t_ind] + lookup_ind_rx[r_ind] - 1; 
                float amp_corr = lookup_amp_tx[t_ind]*lookup_amp_rx[r_ind]*tt_weight[ii];
                if(index<0){
                    }
                else if(index>n){
                    }
                else
                    {
                    int set_val = ii*(n)+index;
                    real = real_exp[set_val];
                    real = real*amp_corr;
                    imag = img_exp[set_val];
                    imag = imag*amp_corr;
                    }
                // sum each val
                tot_real += real;
                tot_imag += imag;
					
            }
		
            // store the final value for the pixel
            //result[pix] = sqrt(tot_real*tot_real + tot_imag*tot_imag);
            real_result[pix] = tot_real;
            imag_result[pix] = tot_imag;
    }
}

__global__ void tfm_near_hmc(float* real_result,float* imag_result,const int n,const int combs, const float* real_exp,const float* img_exp,const int* transmit,const int* receive,const int* lookup_ind_tx, const int* lookup_ind_rx,const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float* lookup_amp_tx,const float* lookup_amp_rx,const float* tt_weight){

	// get pixel's coordinates
    int pix = blockIdx.x*blockDim.x+threadIdx.x;
     
    if (pix<tot_pix){
            //local variable
            float tot_real = 0, tot_imag = 0;

            for(int ii = 0; ii < combs; ii++){
                float real = 0;
                float imag = 0;
                int tx = transmit[ii]-1;
                int rx = receive[ii]-1;
                int t_ind = (tx*grid_x*grid_y*grid_z)+pix;
                int r_ind = (rx*grid_x*grid_y*grid_z)+pix;

                int index1 = lookup_ind_tx[t_ind] + lookup_ind_rx[r_ind]-1; 
                float amp_corr1 = lookup_amp_tx[t_ind]*lookup_amp_rx[r_ind]*tt_weight[ii]/2;
                int index2 = lookup_ind_tx[r_ind] + lookup_ind_rx[t_ind]-1; 
                float amp_corr2 = lookup_amp_tx[r_ind]*lookup_amp_rx[t_ind]*tt_weight[ii]/2;

                if(index1<0){
                    }
                else if(index1>n){
                    }
                else
                    {
                    int set_val1 = ii*(n)+index1;
                    real = real_exp[set_val1]*amp_corr1;
                    imag = img_exp[set_val1]*amp_corr1;
                    // sum each val
                    tot_real += real;
                    tot_imag += imag;
                    }
                real = 0;
                imag = 0;
                if(index2<0){
                    }
                else if(index2>n){
                    }
                else
                    {
                    int set_val2 = ii*(n)+index2;
                    real = real_exp[set_val2]*amp_corr2;
                    imag = img_exp[set_val2]*amp_corr2;
                    // sum each val
                    tot_real += real;
                    tot_imag += imag;
                    }
                
					
            }
		
            // store the final value for the pixel
            //result[pix] = sqrt(tot_real*tot_real + tot_imag*tot_imag);
            real_result[pix] = tot_real;
            imag_result[pix] = tot_imag;
    }
}

__global__ void tfm_linear_norm(float* real_result,float* imag_result,const int n,const int combs, const float* real_exp,const float* img_exp,const int* transmit,const int* receive,const float* lookup_time,const float* time, const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float* lookup_amp, const float* tt_weight){

	// get pixel's coordinates
    int pix = blockIdx.x*blockDim.x+threadIdx.x;
   
        if (pix<tot_pix){
            //local variable
            float tot_real = 0, tot_imag = 0;
            float dt = time[1]-time[0];

            for(int ii = 0; ii < combs; ii++){
                float real = 0;
                float imag = 0;
                int tx = transmit[ii]-1;
                int rx = receive[ii]-1;
                int t_ind = (tx*grid_x*grid_y*grid_z)+pix;
                int r_ind = (rx*grid_x*grid_y*grid_z)+pix;

                float time_val = lookup_time[t_ind] + lookup_time[r_ind]; 
                float amp_corr = lookup_amp[t_ind]*lookup_amp[r_ind]*tt_weight[ii];
                float time_diff = time_val-time[0];
                if(time_diff<0){
                    }
                else if(time_val > time[n-1]){
                    }
                else
                    {
                    int time_0 = floorf((time_val-time[0])/dt);
                    int set_val = ii*(n)+time_0;
                    float real_y1 = real_exp[set_val];
                    float imag_y1 = img_exp[set_val];
                    float real_y2 = real_exp[set_val+1];
                    float imag_y2 = img_exp[set_val+1];

                    float real_dy = real_y2-real_y1;
                    float imag_dy = imag_y2-imag_y1;
    
                    real = real_y1+real_dy*(time_val-time[time_0])/dt;
                    real = real*amp_corr;
                    imag = imag_y1+imag_dy*(time_val-time[time_0])/dt;
                    imag = imag*amp_corr;
                    }
                // sum each val
                tot_real += real;
                tot_imag += imag;
                    
			}
		
		// store the final value for the pixel
		real_result[pix] = tot_real;
        imag_result[pix] = tot_imag;
    }
}

__global__ void tfm_linear_2dly(float* real_result,float* imag_result,const int n,const int combs, const float* real_exp,const float* img_exp,const int* transmit,const int* receive,const float* lookup_time_tx,const float* lookup_time_rx,const float* time, const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float* lookup_amp_tx, const float* lookup_amp_rx, const float* tt_weight){

	// get pixel's coordinates
    int pix = blockIdx.x*blockDim.x+threadIdx.x;
   
        if (pix<tot_pix){
            //local variable
            float tot_real = 0, tot_imag = 0;
            float dt = time[1]-time[0];

            for(int ii = 0; ii < combs; ii++){
                float real = 0;
                float imag = 0;
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
                    int time_0 = floorf((time_val-time[0])/dt);
                    int set_val = ii*(n)+time_0;
                    float real_y1 = real_exp[set_val];
                    float imag_y1 = img_exp[set_val];
                    float real_y2 = real_exp[set_val+1];
                    float imag_y2 = img_exp[set_val+1];

                    float real_dy = real_y2-real_y1;
                    float imag_dy = imag_y2-imag_y1;
    
                    real = real_y1+real_dy*(time_val-time[time_0])/dt;
                    real = real*amp_corr;
                    imag = imag_y1+imag_dy*(time_val-time[time_0])/dt;
                    imag = imag*amp_corr;
                    }
                // sum each val
                tot_real += real;
                tot_imag += imag;
                    
			}
		
		// store the final value for the pixel
		real_result[pix] = tot_real;
        imag_result[pix] = tot_imag;
    }
}

__global__ void tfm_linear_hmc(float* real_result,float* imag_result,const int n,const int combs, const float* real_exp,const float* img_exp,const int* transmit,const int* receive,const float* lookup_time_tx,const float* lookup_time_rx,const float* time, const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float* lookup_amp_tx, const float* lookup_amp_rx, const float* tt_weight){

	// get pixel's coordinates
    int pix = blockIdx.x*blockDim.x+threadIdx.x;
   
        if (pix<tot_pix){
            //local variable
            float tot_real = 0, tot_imag = 0;
            float dt = time[1]-time[0];

            for(int ii = 0; ii < combs; ii++){
                float real = 0;
                float imag = 0;
                int tx = transmit[ii]-1;
                int rx = receive[ii]-1;
                int t_ind = (tx*grid_x*grid_y*grid_z)+pix;
                int r_ind = (rx*grid_x*grid_y*grid_z)+pix;

                float time_val1 = lookup_time_tx[t_ind] + lookup_time_rx[r_ind]; 
                float time_val2 = lookup_time_tx[r_ind] + lookup_time_rx[t_ind]; 
                float amp_corr1 = lookup_amp_tx[t_ind]*lookup_amp_rx[r_ind]*tt_weight[ii]/2;
                float amp_corr2 = lookup_amp_tx[r_ind]*lookup_amp_rx[t_ind]*tt_weight[ii]/2;
                float time_diff1 = time_val1-time[0];
                float time_diff2 = time_val2-time[0];
                if(time_diff1<0){
                    }
                else if(time_val1 > time[n-1]){
                    }
                else
                    {
                    int time_0 = floorf((time_val1-time[0])/dt);
                    int set_val = ii*(n)+time_0;
                    float real_y1 = real_exp[set_val];
                    float imag_y1 = img_exp[set_val];
                    float real_y2 = real_exp[set_val+1];
                    float imag_y2 = img_exp[set_val+1];

                    float real_dy = real_y2-real_y1;
                    float imag_dy = imag_y2-imag_y1;
    
                    real = real_y1+real_dy*(time_val1-time[time_0])/dt;
                    real = real*amp_corr1;
                    imag = imag_y1+imag_dy*(time_val1-time[time_0])/dt;
                    imag = imag*amp_corr1;
                    // sum each val
                    tot_real += real;
                    tot_imag += imag;
                    }
               
                real = 0;
                imag = 0;
                if(time_diff2<0){
                    }
                else if(time_val2 > time[n-1]){
                    }
                else
                    {
                    int time_0 = floorf((time_val2-time[0])/dt);
                    int set_val = ii*(n)+time_0;
                    float real_y1 = real_exp[set_val];
                    float imag_y1 = img_exp[set_val];
                    float real_y2 = real_exp[set_val+1];
                    float imag_y2 = img_exp[set_val+1];

                    float real_dy = real_y2-real_y1;
                    float imag_dy = imag_y2-imag_y1;
    
                    real = real_y1+real_dy*(time_val2-time[time_0])/dt;
                    real = real*amp_corr2;
                    imag = imag_y1+imag_dy*(time_val2-time[time_0])/dt;
                    imag = imag*amp_corr2;
                    // sum each val
                    tot_real += real;
                    tot_imag += imag;
                    }

                    
			}
		
		// store the final value for the pixel
		real_result[pix] = tot_real;
        imag_result[pix] = tot_imag;
    }
}


__device__ float lanczos_window2(const float t, const float a)
{
    if (abs(t) < 1e-7)
    {   
        return 1.0F;
    }
    else
    {
        float pi=asin(1.0F)*2.0F;
        float pix=pi*t;
        return a*sin(pix)*sin(pix/a)/(pix*pix); //*sinc_normalised(t)*sinc_normalised(t/a);
    }
}

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



__global__ void tfm_lanczos_norm(float* real_result,float* imag_result,const int n,const int combs, const float* real_exp,const float* img_exp,const int* transmit,const int* receive,const float* lookup_time,const float* time, const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float* lookup_amp,const float* tt_weight, const float aFactor){

/*
 *    Delay and sum with Lanczos interpolation with factor 'a' of the scanlines.
 *
 *    https://en.wikipedia.org/wiki/Lanczos_resampling
 */

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

                float time_val = lookup_time[t_ind] + lookup_time[r_ind]; 
                float amp_corr = lookup_amp[t_ind]*lookup_amp[r_ind]*tt_weight[ii];
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


__global__ void tfm_lanczos_2dly(float* real_result,float* imag_result,const int n,const int combs, const float* real_exp,const float* img_exp,const int* transmit,const int* receive,const float* lookup_time_tx,const float* lookup_time_rx,const float* time, const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float* lookup_amp_tx, const float* lookup_amp_rx, const float* tt_weight, const float aFactor){

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


__global__ void tfm_lanczos_hmc(float* real_result,float* imag_result,const int n,const int combs, const float* real_exp,const float* img_exp,const int* transmit,const int* receive,const float* lookup_time_tx,const float* lookup_time_rx,const float* time, const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float* lookup_amp_tx, const float* lookup_amp_rx, const float* tt_weight,const float aFactor){

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
                int tx = transmit[ii]-1;
                int rx = receive[ii]-1;
                int t_ind = (tx*grid_x*grid_y*grid_z)+pix;
                int r_ind = (rx*grid_x*grid_y*grid_z)+pix;

                float time_val1 = lookup_time_tx[t_ind] + lookup_time_rx[r_ind]; 
                float time_val2 = lookup_time_tx[r_ind] + lookup_time_rx[t_ind]; 
                float amp_corr1 = lookup_amp_tx[t_ind]*lookup_amp_rx[r_ind]*tt_weight[ii]/2;
                float amp_corr2 = lookup_amp_tx[r_ind]*lookup_amp_rx[t_ind]*tt_weight[ii]/2;
                float time_diff1 = time_val1-time[0];
                float time_diff2 = time_val2-time[0];
                if(time_diff1<0){
                    }
                else if(time_val1 > time[n-1]){
                    }
                else
                    {
                    int scanline = ii*(n);
                    float lookup_index_float = time_diff1 * invdt;
                    // sum each val
                    tot_real += amp_corr1*lanczos_interpolation2(&real_exp[scanline],lookup_index_float,n,aFactor,lanczos_window2,NLanczos2);
                    tot_imag += amp_corr1*lanczos_interpolation2(&img_exp[scanline],lookup_index_float,n,aFactor,lanczos_window2,NLanczos2);
                    //tot_real += amp_corr1*lanczos_interpolation(&real_exp[scanline],lookup_index_float,n,aFactor);
                    //tot_imag += amp_corr1*lanczos_interpolation(&img_exp[scanline],lookup_index_float,n,aFactor);
                    }
                if(time_diff2<0){
                    }
                else if(time_val2 > time[n-1]){
                    }
                else
                    {
                    int scanline = ii*(n);
                    float lookup_index_float = time_diff2 * invdt;
                    // sum each val
                    tot_real += amp_corr2*lanczos_interpolation2(&real_exp[scanline],lookup_index_float,n,aFactor,lanczos_window2,NLanczos2);
                    tot_imag += amp_corr2*lanczos_interpolation2(&img_exp[scanline],lookup_index_float,n,aFactor,lanczos_window2,NLanczos2);
                    //tot_real += amp_corr2*lanczos_interpolation(&real_exp[scanline],lookup_index_float,n,aFactor);
                    //tot_imag += amp_corr2*lanczos_interpolation(&img_exp[scanline],lookup_index_float,n,aFactor);
                    }

                    
			}
		
		// store the final value for the pixel
		real_result[pix] = tot_real;
        imag_result[pix] = tot_imag;
    }
}