__global__ void tfm_3d(float* real_result,float* imag_result,const int n,
                                        const int combs, const float* real_exp,const float* img_exp,
                                            const int* transmit,const int* receive,const float vel,const float* time,
                                             const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float x_min,
                                                 const float z_min, const float y_min, const float dx, const float* el_x,const float* el_y){

	// get pixel's coordinates
    int pix = blockIdx.x*blockDim.x+threadIdx.x;
   
        if (pix<tot_pix){
            //local variable
            float tot_real = 0, tot_imag = 0, amp_corr = 0;
            float x_val = 0, y_val = 0, z_val = 0, t_dist = 0, r_dist = 0;
            float dt = time[1]-time[0];
            int x_loc = 0, y_loc = 0, z_loc = 0;
            //if (grid_z == 1){
            //    z_loc = pix-grid_x*floorf(pix / grid_x);
            //    x_loc = floorf(pix / grid_x);
            //    y_loc = 0;
            //    }
            //else
            //    {
                y_loc = pix-grid_y*floorf(pix / grid_y);
                z_loc = floorf(pix / (grid_x * grid_y));
                int temp_pix = pix - z_loc * grid_x * grid_y;
                x_loc = floorf(temp_pix / grid_y);
                y_val = y_min + y_loc*dx;
            //    }

            x_val = x_min + x_loc*dx;            
            z_val = z_min + z_loc*dx;
            
            for(int ii = 0; ii < combs; ii++){
                float real = 0;
                float imag = 0;
                int tx = transmit[ii]-1;
                int rx = receive[ii]-1;
                float t_x_comp = powf((el_x[tx]-x_val),2);
                float z_comp = powf(z_val,2);
                float r_x_comp = powf((el_x[rx]-x_val),2);
                
                //if (grid_z ==1){
                //    t_dist = t_x_comp + z_comp;
                //    r_dist = r_x_comp + z_comp;
                //    }
                //else
                //    {
                    float t_y_comp = powf((el_y[tx]-y_val),2);
                    float r_y_comp = powf((el_y[rx]-y_val),2);
                    t_dist = t_x_comp + t_y_comp + z_comp;
                    r_dist = r_x_comp + r_y_comp + z_comp;
                //    }
                t_dist=sqrtf(t_dist);
                r_dist=sqrtf(r_dist);

                float time_val = (t_dist + r_dist)/vel; 
                
                if( tx == rx){
                    amp_corr = 1;
                    }
                else
                    {
                    amp_corr = 2;
                    }

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

__global__ void tfm_2d(float* real_result,float* imag_result,const int n,
                                        const int combs, const float* real_exp,const float* img_exp,
                                            const int* transmit,const int* receive,const float vel,const float* time,
                                             const int tot_pix, const int grid_x, const int grid_y, const int grid_z, const float x_min,
                                                 const float z_min, const float y_min, const float dx, const float* el_x,const float* el_y){

	// get pixel's coordinates
    int pix = blockIdx.x*blockDim.x+threadIdx.x;
   
        if (pix<tot_pix){
            //local variable
            float tot_real = 0, tot_imag = 0, amp_corr = 0;
            float x_val = 0, y_val = 0, z_val = 0, t_dist = 0, r_dist = 0;
            float dt = time[1]-time[0];
            int x_loc = 0, y_loc = 0, z_loc = 0;
            //if (grid_z == 1){
                z_loc = pix-grid_x*floorf(pix / grid_x);
                x_loc = floorf(pix / grid_x);
                y_loc = 0;
            //    }
            //else
            //    {
            //    y_loc = pix-grid_y*floorf(pix / grid_y);
            //    z_loc = floorf(pix / (grid_x * grid_y));
            //    int temp_pix = pix - z_loc * grid_x * grid_y;
            //    x_loc = floorf(temp_pix / grid_y);
            //    y_val = y_min + y_loc*dx;
            //    }

            x_val = x_min + x_loc*dx;            
            z_val = z_min + z_loc*dx;
            
            for(int ii = 0; ii < combs; ii++){
                float real = 0;
                float imag = 0;
                int tx = transmit[ii]-1;
                int rx = receive[ii]-1;
                float t_x_comp = powf((el_x[tx]-x_val),2);
                float z_comp = powf(z_val,2);
                float r_x_comp = powf((el_x[rx]-x_val),2);
                
                //if (grid_z ==1){
                    t_dist = t_x_comp + z_comp;
                    r_dist = r_x_comp + z_comp;
                //    }
                //else
                //    {
                //    float t_y_comp = powf((el_y[tx]-y_val),2);
                //    float r_y_comp = powf((el_y[rx]-y_val),2);
                //    t_dist = t_x_comp + t_y_comp + z_comp;
                //    r_dist = r_x_comp + r_y_comp + z_comp;
                //    }
                t_dist=sqrtf(t_dist);
                r_dist=sqrtf(r_dist);

                float time_val = (t_dist + r_dist)/vel; 
                
                if( tx == rx){
                    amp_corr = 1;
                    }
                else
                    {
                    amp_corr = 2;
                    }

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