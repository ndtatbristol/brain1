__global__ void gpu_imm_foc(float* foc_law,float* foc_amp, const int grid_x, const int grid_z,const int grid_surf,const float* s_fine_x,const float* s_fine_z,const float* s_coars_z, const float* mesh_x, const float* mesh_z, const float* x_arr, const float* z_arr, const float couple_vel, const float mat_vel, const int num_els){

	// get pixel's coordinates
    int pix = blockIdx.x*blockDim.x+threadIdx.x;
    int img_pix = grid_x * grid_z;
    int tot_pix = img_pix * num_els;
   
    if (pix<tot_pix){
            //local variable
            float couple_time = 0, mat_time = 0, tot_time = 0, old_time = 1, dist = 0, coupl_ang = 0;
            int el = floorf(pix/img_pix);
            int img_loc = pix - (img_pix * el);
            int x_loc = floorf(img_loc/grid_x);
            if (s_coars_z[x_loc] < mesh_z[img_loc]){
                for(int ii = 0; ii < grid_surf; ii++){
                    couple_time = powf((x_arr[el]-s_fine_x[ii]),2) + powf((z_arr[el]-s_fine_z[ii]),2);
                    mat_time = powf((s_fine_x[ii]-mesh_x[img_loc]),2) + powf((s_fine_z[ii]-mesh_z[img_loc]),2);
                    tot_time = sqrtf(couple_time)/couple_vel + sqrtf(mat_time)/mat_vel;
                    if (tot_time < old_time){
                        old_time = tot_time;
                        dist = sqrtf(mat_time)+sqrtf(couple_time);
                        coupl_ang = atan2f((x_arr[el]-s_fine_x[ii]),(z_arr[el]-s_fine_z[ii]));
                        }
                    }               
                foc_amp[pix] = 1;// / dist * cosf(coupl_ang) * sqrtf(dist);
                //foc_amp[pix] = 1 / dist * sqrtf(dist);
                foc_law[pix] = old_time;
            }
    }
}