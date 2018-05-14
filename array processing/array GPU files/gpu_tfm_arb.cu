__global__ void gpu_tfm(float* real_result,float* imag_result,const int n,const int combs, const float* real_exp,const float* img_exp,const int* transmit,const int* receive,const int* lookup_ind, const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float* lookup_amp, const float* tt_weight){

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