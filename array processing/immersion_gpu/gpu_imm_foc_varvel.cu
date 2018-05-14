__global__ void gpu_imm_foc_varvel(float* foc_law,float* foc_amp, const int grid_x, const int grid_z,const int grid_surf,const float* s_fine_x,const float* s_fine_z,const float* s_coars_z, const float* mesh_x, const float* mesh_z, const float* x_arr, const float* z_arr, const float couple_vel, const float mat_vel, const int num_els, const float* mu, const int poly_order, const float* poly, const float ang_lim, const float atten){

	// get pixel's coordinates
    int pix = blockIdx.x*blockDim.x+threadIdx.x;
    int img_pix = grid_x * grid_z;
    int tot_pix = img_pix * num_els;
   
    if (pix<tot_pix){
            //local variable
            float couple_dist = 0, mat_dist = 0, tot_time = 0, old_time = 1, dist = 0, couple_ang = 0;
            
            int el = floorf(pix/img_pix);
            int img_loc = pix - (img_pix * el);
            int x_loc = floorf(img_loc/grid_x);
            if (s_coars_z[x_loc] < mesh_z[img_loc]){
                float mat_ang = 0;
                for(int ii = 0; ii < grid_surf; ii++){
                    float mat_vel = 2900, scaled_x = (mat_ang-mu[0])/mu[1];
                    couple_dist = sqrtf(powf((x_arr[el]-s_fine_x[ii]),2) + powf((z_arr[el]-s_fine_z[ii]),2));
                    mat_dist = sqrtf(powf((s_fine_x[ii]-mesh_x[img_loc]),2) + powf((s_fine_z[ii]-mesh_z[img_loc]),2));
                    //int jj = 0;
                    //for(jj = 0; jj <= poly_order; jj++){
                    //    mat_vel = poly[jj]*powf(scaled_x,poly_order-jj) + mat_vel;
                    //    }
                    tot_time = couple_dist/couple_vel + mat_dist/mat_vel;
                    if (tot_time < old_time){
                        mat_ang = atan2f((mesh_x[img_loc]-s_fine_x[ii]),(mesh_z[img_loc]-s_fine_z[ii]));
                        mat_ang = fabsf(mat_ang);
                        old_time = tot_time;
                        couple_ang = atan2f((s_fine_x[ii]-x_arr[el]),(s_fine_z[ii]-z_arr[el]));
                        dist = couple_dist + mat_dist;
                        
                        }
                    }
                
                if (mat_ang <= ang_lim){
                    foc_amp[pix] = 1 / dist * cosf(couple_ang) * sqrtf(dist) * expf( (mesh_z[img_loc]-s_coars_z[x_loc]) * atten / 2);
                    //foc_amp[pix] = 1;
                    foc_law[pix] = old_time;
                    }
                else {
                    foc_law[pix] = old_time;
                    }
            }
    }
}