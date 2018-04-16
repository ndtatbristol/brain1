__global__ void gpu_tfm_linear_arb(float* real_result,float* imag_result,const int n,const int combs, const float* real_exp,const float* img_exp,const int* transmit,const int* receive,const float* lookup_time,const float* time, const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float* lookup_amp){

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
                float amp_corr = lookup_amp[t_ind]*lookup_amp[r_ind];
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