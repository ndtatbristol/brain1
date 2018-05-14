__global__ void gpu_tfm_arb_2dly_hmc(float* real_result,float* imag_result,const int n,const int combs, const float* real_exp,const float* img_exp,const int* transmit,const int* receive,const int* lookup_ind_tx, const int* lookup_ind_rx,const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float* lookup_amp_tx,const float* lookup_amp_rx,const float* tt_weight){

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