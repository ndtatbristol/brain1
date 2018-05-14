__global__ void gpu_tfm_linear_arb_2dly_hmc(float* real_result,float* imag_result,const int n,const int combs, const float* real_exp,const float* img_exp,const int* transmit,const int* receive,const float* lookup_time_tx,const float* lookup_time_rx,const float* time, const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float* lookup_amp_tx, const float* lookup_amp_rx, const float* tt_weight){

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