#include "mex.h"
#include <omp.h>
#include <math.h>
/*
 * based upon xtimesy.c - example found in API guide
 */

// COMPILE WITH : (Visual Studio 2013 tested)
// mex mex_lookup_times8SP.c CFLAGS="\$CFLAGS /openmp" LDFLAGS="\$LDFLAGS /openmp" COMPFLAGS="/openmp $COMPFLAGS"

float travel_time(float x1,float y1,float x2,float y2,float vel_water_material_ratio,float intX)
{
    float d1=sqrt((x1-intX)*(x1-intX)+(y1*y1));
    float d2=sqrt((x2-intX)*(x2-intX)+(y2*y2));
    float dxx=d1*vel_water_material_ratio+d2;
    return dxx;
}

float travel_time3d(float x1,float y1, float z1, float x2,float y2,float z2,float vel_water_material_ratio,float intX, float intY)
{
    float d1=sqrt((x1-intX)*(x1-intX)+(y1-intY)*(y1-intY)+z1*z1);
    float d2=sqrt((x2-intX)*(x2-intX)+(y2-intY)*(y2-intY)+z2*z2);
    float dxx=d1*vel_water_material_ratio+d2;
    return dxx;
}

float travel_time_leg2d(float x1,float y1,float x2,float y2,float inv_vel)
{
    float d1=sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2));
    return d1*inv_vel;
}

float travel_time_leg3d(float x1,float y1,float z1, float x2,float y2,float z2,float inv_vel)
{
    float d1=sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2)+(z1-z2)*(z1-z2));
    return d1*inv_vel;
}

/* the gateway function */

//Format is
//[result] = function ( path_opt , dimension_opt, exp_data.tx, exp_data.rx, exp_data.time, lookup_times)

void mexFunction( int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[])
{
    const float *probe_coordsX, *probe_coordsZ,*probe_coordsY, *pixel_coordsX, *pixel_coordsZ, *pixel_coordsY;
    const float *frontwall_coordsX, *frontwall_coordsZ, *frontwall_coordsY, *backwall_coordsX, *backwall_coordsZ, *backwall_coordsY;
    float *lookup_times;
    int *dim_array;
    int CHUNKSIZE=4;
    int CHUNKSIZE2=50;
    
    int nelements=0,npixels=0, nfrontwall=0, nbackwall=0;
    int iPixel, iElement;
    float interface_pointX,interface_pointY;
    float *travel_times_fb, *travel_times_pf, *travel_times_bp,*cur_time;
    int *searchOrder, *iParent1, *iParent2;
    
    //Setup options
    if(nrhs<3) {mexErrMsgIdAndTxt( "MATLAB:invalidNumInputs","need 3 inputs to define options: path_opt,dimension_opt,direction_opt");}
    int path_opt=(int)mxGetScalar(prhs[0]);         //Solver and setup type (i.e. contact, probe->frontwall->pixel or probe->frontwall->backwall->pixel)
    //Option    = 0: Contact (probe->pixel)
    //          = 1: probe->frontwall->pixel (with Newton-Raphson Solver, only applies for 1D otherwise will default to below option)
    //          = 2: probe->frontwall->pixel (discretised frontwall search - fermat solver)
    //          = 3: probe->frontwall->backwall->pixel (discretised frontwall & backwall - fermat solver)
    int dimension_opt=(int)mxGetScalar(prhs[1]);    //Allowed values: 2 = 2D, 3 = 3D
    int direction_opt=(int)mxGetScalar(prhs[2]);    //Is direction probe->pixel (typical if pixel>elements) = 0 or vice versa (when pixel count is less than element count) = 1

    //Check options specified
    if (direction_opt < 0 || direction_opt > 1)
    {
        mexErrMsgIdAndTxt( "MATLAB:invalidInput","Direction option must be 0 (pixel dominated) or 1 (element dominated)");
    }
    if (dimension_opt < 2 || dimension_opt > 3)
    {
        mexErrMsgIdAndTxt( "MATLAB:invalidInput","Dimension option must be 2 (2D) or 3 (for 3D)");
    }
    if (path_opt < 0 || path_opt > 3)
    {
        mexErrMsgIdAndTxt( "MATLAB:invalidInput","Path option must be between 0 (contact), (1,2 immersed, direct) and 3 (skip paths)");
    }
//     if (path_opt == 1 && dimension_opt == 3)
//     {
//         path_opt=2; //Cannot use Newton-Raphson if 3D, change to Fermat Solver
//     }
    
    int irhs=2;
    //Probe Coordinates
    if(nrhs<irhs+1+dimension_opt) {mexErrMsgIdAndTxt( "MATLAB:invalidNumInputs","Cannot find probe coordinate inputs");}
    irhs+=1; probe_coordsX = (float *)mxGetPr(prhs[irhs]);
    dim_array=mxGetDimensions(prhs[irhs]);
    nelements= max(dim_array[0],dim_array[1]);
    if (dimension_opt>2)
    {
        irhs+=1; probe_coordsY = (float *)mxGetPr(prhs[irhs]);
    }
    irhs+=1; probe_coordsZ = (float *)mxGetPr(prhs[irhs]);
    
    //Pixel Coordinates
    if(nrhs<irhs+1+dimension_opt) {mexErrMsgIdAndTxt( "MATLAB:invalidNumInputs","Cannot find pixel coordinate inputs");}
    irhs+=1; pixel_coordsX = (float *)mxGetPr(prhs[irhs]);
    dim_array=mxGetDimensions(prhs[irhs]);
    npixels= max(dim_array[0],dim_array[1]);
    if (dimension_opt>2)
    {
        irhs+=1; pixel_coordsY = (float *)mxGetPr(prhs[irhs]);
    }
    irhs+=1; pixel_coordsZ = (float *)mxGetPr(prhs[irhs]);
    int CHUNKSIZE_PIXEL=max(1,(int)(npixels/16.0+0.5));
    CHUNKSIZE_PIXEL=min(CHUNKSIZE_PIXEL,1000);        
    //Frontwall Coordinates
    if (path_opt>1)
    {
        if(nrhs<irhs+1+dimension_opt) {mexErrMsgIdAndTxt( "MATLAB:invalidNumInputs","Cannot find frontwall coordinate inputs");}
        irhs+=1; frontwall_coordsX = (float *)mxGetPr(prhs[irhs]);
        dim_array=mxGetDimensions(prhs[irhs]);
        nfrontwall= max(dim_array[0],dim_array[1]);
        if (dimension_opt>2)
        {
            irhs+=1; frontwall_coordsY = (float *)mxGetPr(prhs[irhs]);
        }
        irhs+=1; frontwall_coordsZ = (float *)mxGetPr(prhs[irhs]);   
    }
    //Backwall Coordinates
    if (path_opt>2)
    {
        if(nrhs<irhs+1+dimension_opt) {mexErrMsgIdAndTxt( "MATLAB:invalidNumInputs","Cannot find backwall coordinate inputs");}
        irhs+=1; backwall_coordsX = (float *)mxGetPr(prhs[irhs]);
        dim_array=mxGetDimensions(prhs[irhs]);
        nbackwall= max(dim_array[0],dim_array[1]);
        if (dimension_opt>2)
        {
            irhs+=1; backwall_coordsY = (float *)mxGetPr(prhs[irhs]);
        }
        irhs+=1; backwall_coordsZ = (float *)mxGetPr(prhs[irhs]);   
    }
    
    /* Display dimensions */
    //mexPrintf("nelements: %d npixels: %d\n", nelements, npixels);
    //mexPrintf("nfrontwall: %d nbackwall: %d\n", nfrontwall, nbackwall);
    
    if(nrhs<irhs+1+4) {mexErrMsgIdAndTxt( "MATLAB:invalidNumInputs","Cannot find 3 velocity & sample freq inputs");}
    irhs+=1; float vel_water=(float)mxGetScalar(prhs[irhs]);
    irhs+=1; float vel_material=(float)mxGetScalar(prhs[irhs]);
    irhs+=1; float vel_material2=(float)mxGetScalar(prhs[irhs]);
    irhs+=1; float sample_freq=(float)mxGetScalar(prhs[irhs]);
    
    if(nrhs<irhs+1+1) {mexErrMsgIdAndTxt( "MATLAB:invalidNumInputs","Cannot find travel time previously calculated option");}
    irhs+=1; int travels_calculated = (int)mxGetScalar(prhs[irhs]); // -1 -> need to calculate, don't store, 0 -> Not yet calculated, store afterwards 1 -> Use previously calculated data
    
    if (direction_opt>0 || path_opt < 2)
    {
        travels_calculated=-1;
    }
    
    // Create output
    int i=-1;
    i+=1; plhs[i] = mxCreateNumericMatrix( (mwSize)nelements, (mwSize)npixels,mxSINGLE_CLASS, mxREAL);
    lookup_times = (float *)mxGetPr(plhs[i]);
    if (travels_calculated == 0)
    {
        if (path_opt>2)
        {
            i+=1; plhs[i] = mxCreateNumericMatrix( (mwSize)nelements*nbackwall, (mwSize)1,mxSINGLE_CLASS, mxREAL);
        }
        else
        {
            i+=1; plhs[i] = mxCreateNumericMatrix( (mwSize)nelements*nfrontwall, (mwSize)1,mxSINGLE_CLASS, mxREAL);
        }
        cur_time = (float *)mxGetPr(plhs[i]);
    }
    else if (travels_calculated>0)
    {
        if(nrhs<irhs+1+1) {mexErrMsgIdAndTxt( "MATLAB:invalidNumInputs","Cannot find travel timse previously calculated");}
        irhs+=1; cur_time=(float *)mxGetPr(prhs[irhs]);
    }

    
    if(nlhs != i+1)
    {
        mexErrMsgIdAndTxt( "MATLAB:invalidNumOutputs","different number of outputs required.");
    }
    
    if (direction_opt>0 && dimension_opt < 3)
    {
        if(nrhs<irhs+1+3) {mexErrMsgIdAndTxt( "MATLAB:invalidNumInputs","Cannot find element search order and tree information");}
        irhs+=1; searchOrder=(int *)mxGetPr(prhs[irhs]);
        irhs+=1; iParent1=(int *)mxGetPr(prhs[irhs]);
        irhs+=1; iParent2=(int *)mxGetPr(prhs[irhs]);
    }
    
    
    //Create temporary arrays for travel time storage for specific legs of path
    if (path_opt>2)
    {
        // Backwall to pixel (or vice versa)
        travel_times_bp=(float*) malloc(npixels*nbackwall*sizeof(float));
        if (travels_calculated<1)
        {
            travel_times_fb=(float*) malloc(nfrontwall*nbackwall*sizeof(float));
            travel_times_pf=(float*) malloc(nfrontwall*nelements*sizeof(float));   
        }
        if (travels_calculated<0)
        {
            cur_time=(float*) malloc(nelements*nbackwall*sizeof(float));   
        }
            
    }
    else if (path_opt>1)
    {
        if (travels_calculated<0)
        {
            cur_time=(float*) malloc(nfrontwall*nelements*sizeof(float));   
        }
    }
    
    
    // Create useful variables for later
    
    //Newton-Raphson related variables
    float vel_water_material_ratio=vel_material/vel_water;
    float small_amount=0.1*vel_water/sample_freq;
    float half_small_amount=0.5*small_amount;
    float inv_diff2=1.0/(2.0*small_amount);
    float inv_diffsqr=1.0/(small_amount*small_amount);
    //Fermat Solver variables
    float inv_vel_material1=1.0/vel_material;
    float inv_vel_material2=1.0/vel_material2;
    float inv_vel_water=1.0/vel_water;
    int iFront, iBackwall;


    // OpenMP Parallel Solve (path_opt/direction_opt/dimension_opt dependent)
    
#pragma omp parallel default (none) firstprivate(iElement,iPixel,interface_pointX,interface_pointY,iFront,iBackwall) shared(half_small_amount,dimension_opt,CHUNKSIZE_PIXEL,direction_opt,cur_time,travels_calculated,searchOrder,iParent1,iParent2,CHUNKSIZE2,nfrontwall,nbackwall,backwall_coordsX,backwall_coordsY,backwall_coordsZ,frontwall_coordsX,frontwall_coordsY,frontwall_coordsZ,travel_times_fb,travel_times_pf,travel_times_bp,inv_vel_material1,inv_vel_material2,inv_vel_water,small_amount,path_opt,inv_diff2,inv_diffsqr,probe_coordsX,probe_coordsY,probe_coordsZ,pixel_coordsX,pixel_coordsY,pixel_coordsZ,lookup_times, CHUNKSIZE,nelements,npixels,vel_water_material_ratio,vel_material,vel_water)
{
    
    //Newton-Raphson temporary arrays for parallel (not yet allocated)
    int *iFrontArray;
    float *cur_time2;
    
    switch (path_opt)
    {
        case 0: //Contact
            if (dimension_opt>2)
            {
#pragma omp for schedule(dynamic,CHUNKSIZE_PIXEL)
                for (iPixel=0; iPixel<npixels;iPixel++)
                {
                    int fb_int=iPixel*nelements;
                    for (iElement=0; iElement<nelements;iElement++)
                    {
                        lookup_times[fb_int+iElement]=travel_time_leg3d(probe_coordsX[iElement],probe_coordsY[iElement],probe_coordsZ[iElement],pixel_coordsX[iPixel],pixel_coordsY[iPixel],pixel_coordsZ[iPixel],inv_vel_water);
                    }
                }
            }
            else
            {
#pragma omp for schedule(dynamic,CHUNKSIZE_PIXEL)
                for (iPixel=0; iPixel<npixels;iPixel++)
                {
                    int fb_int=iPixel*nelements;
                    for (iElement=0; iElement<nelements;iElement++)
                    {
                        lookup_times[fb_int+iElement]=travel_time_leg2d(probe_coordsX[iElement],probe_coordsZ[iElement],pixel_coordsX[iPixel],pixel_coordsZ[iPixel],inv_vel_water);
                    }
                }                
            }
            break;
        case 1: //Immersed, direct (Newton-Raphson) 
            if (dimension_opt>2)
            {
#pragma omp for schedule(dynamic,CHUNKSIZE_PIXEL)
                for (iPixel=0; iPixel<npixels;iPixel++)
                {
                    for (iElement=0; iElement<nelements;iElement++)
                    {
                        interface_pointX = probe_coordsX[iElement];
                        interface_pointY = probe_coordsY[iElement];
                        //Newton-Raphson Iteration (seeking root to f'(x)=0)
                        for (int jj=0; jj<100; jj++)
                        {
                            float f_x_y=travel_time3d(probe_coordsX[iElement],probe_coordsY[iElement],probe_coordsZ[iElement],pixel_coordsX[iPixel],pixel_coordsY[iPixel],pixel_coordsZ[iPixel],vel_water_material_ratio,interface_pointX,interface_pointY);
                            //central difference
                            float x_plus=interface_pointX+small_amount;
                            float x_half_plus=interface_pointX+half_small_amount;
                            float x_minus=interface_pointX-small_amount;
                            float x_half_minus=interface_pointX-half_small_amount;
                            float y_plus=interface_pointY+small_amount;
                            float y_half_plus=interface_pointY+half_small_amount;
                            float y_minus=interface_pointY-small_amount;
                            float y_half_minus=interface_pointY-half_small_amount;
                            float f_xplushalf_yplushalf=travel_time3d(probe_coordsX[iElement],probe_coordsY[iElement],probe_coordsZ[iElement],pixel_coordsX[iPixel],pixel_coordsY[iPixel],pixel_coordsZ[iPixel],vel_water_material_ratio,x_half_plus,y_half_plus);
                            float f_xminushalf_yplushalf=travel_time3d(probe_coordsX[iElement],probe_coordsY[iElement],probe_coordsZ[iElement],pixel_coordsX[iPixel],pixel_coordsY[iPixel],pixel_coordsZ[iPixel],vel_water_material_ratio,x_half_minus,y_half_plus);
                            float f_xplushalf_yminushalf=travel_time3d(probe_coordsX[iElement],probe_coordsY[iElement],probe_coordsZ[iElement],pixel_coordsX[iPixel],pixel_coordsY[iPixel],pixel_coordsZ[iPixel],vel_water_material_ratio,x_half_plus,y_half_minus);
                            float f_xminushalf_yminushalf=travel_time3d(probe_coordsX[iElement],probe_coordsY[iElement],probe_coordsZ[iElement],pixel_coordsX[iPixel],pixel_coordsY[iPixel],pixel_coordsZ[iPixel],vel_water_material_ratio,x_half_minus,y_half_minus);
                            float fxy=(f_xplushalf_yplushalf-f_xminushalf_yplushalf-f_xplushalf_yminushalf+f_xminushalf_yminushalf)*inv_diffsqr;
                            
                            float f_xplus_y=travel_time3d(probe_coordsX[iElement],probe_coordsY[iElement],probe_coordsZ[iElement],pixel_coordsX[iPixel],pixel_coordsY[iPixel],pixel_coordsZ[iPixel],vel_water_material_ratio,x_plus,interface_pointY);
                            float f_xminus_y=travel_time3d(probe_coordsX[iElement],probe_coordsY[iElement],probe_coordsZ[iElement],pixel_coordsX[iPixel],pixel_coordsY[iPixel],pixel_coordsZ[iPixel],vel_water_material_ratio,x_minus,interface_pointY);
                            float f_xplushalf_y=travel_time3d(probe_coordsX[iElement],probe_coordsY[iElement],probe_coordsZ[iElement],pixel_coordsX[iPixel],pixel_coordsY[iPixel],pixel_coordsZ[iPixel],vel_water_material_ratio,x_half_plus,interface_pointY);
                            float f_xminushalf_y=travel_time3d(probe_coordsX[iElement],probe_coordsY[iElement],probe_coordsZ[iElement],pixel_coordsX[iPixel],pixel_coordsY[iPixel],pixel_coordsZ[iPixel],vel_water_material_ratio,x_half_minus,interface_pointY);

                            float fx=(f_xplushalf_y-f_xminushalf_y)/small_amount;
                            float fxx=(f_xplus_y-2.0*f_x_y+f_xminus_y)*inv_diffsqr;
                            
                            float f_x_yplus=travel_time3d(probe_coordsX[iElement],probe_coordsY[iElement],probe_coordsZ[iElement],pixel_coordsX[iPixel],pixel_coordsY[iPixel],pixel_coordsZ[iPixel],vel_water_material_ratio,interface_pointX,y_plus);
                            float f_x_yminus=travel_time3d(probe_coordsX[iElement],probe_coordsY[iElement],probe_coordsZ[iElement],pixel_coordsX[iPixel],pixel_coordsY[iPixel],pixel_coordsZ[iPixel],vel_water_material_ratio,interface_pointX,y_minus);
                            float f_x_yplushalf=travel_time3d(probe_coordsX[iElement],probe_coordsY[iElement],probe_coordsZ[iElement],pixel_coordsX[iPixel],pixel_coordsY[iPixel],pixel_coordsZ[iPixel],vel_water_material_ratio,interface_pointX,y_half_plus);
                            float f_x_yminushalf=travel_time3d(probe_coordsX[iElement],probe_coordsY[iElement],probe_coordsZ[iElement],pixel_coordsX[iPixel],pixel_coordsY[iPixel],pixel_coordsZ[iPixel],vel_water_material_ratio,interface_pointX,y_half_minus);

                            float fy=(f_x_yplushalf-f_x_yminushalf)/small_amount;
                            float fyy=(f_x_yplus-2.0*f_x_y+f_x_yminus)*inv_diffsqr;
                            
                            float invdetA=1.0/(fxx*fyy-fxy*fxy);
                            
                            float xchange=invdetA*(fx*fyy-fxy*fy);
                            float ychange=invdetA*(-fx*fxy+fxx*fy);
                            //mexPrintf("Pixel %d Element %d f' %e f'' %e changeX: %e fx %e\n", iPixel, iElement, fdash,ffloatdash,xchange,fx);

                            float tchange=sqrt(xchange*xchange+ychange*ychange);
                            if (fabs(tchange) < 1.0e-09)
                            {
                                lookup_times[iPixel*nelements+iElement]=f_x_y/vel_material;
                                break;
                            }
                            else
                            {
                                interface_pointX=interface_pointX-xchange;
                                interface_pointY=interface_pointY-ychange;
                            }
                            
                        }

                    }
                }                
            }
            else
            {
#pragma omp for schedule(dynamic,CHUNKSIZE_PIXEL)
                for (iPixel=0; iPixel<npixels;iPixel++)
                {
                    for (iElement=0; iElement<nelements;iElement++)
                    {
                        interface_pointX = probe_coordsX[iElement];
                        //Newton-Raphson Iteration (seeking root to f'(x)=0)
                        for (int jj=0; jj<100; jj++)
                        {
                            float fx=travel_time(probe_coordsX[iElement],probe_coordsZ[iElement],pixel_coordsX[iPixel],pixel_coordsZ[iPixel],vel_water_material_ratio,interface_pointX);
                            //central difference
                            float x_plus=interface_pointX+small_amount;
                            float x_minus=interface_pointX-small_amount;
                            float fx_plus=travel_time(probe_coordsX[iElement],probe_coordsZ[iElement],pixel_coordsX[iPixel],pixel_coordsZ[iPixel],vel_water_material_ratio,x_plus);
                            float fx_minus=travel_time(probe_coordsX[iElement],probe_coordsZ[iElement],pixel_coordsX[iPixel],pixel_coordsZ[iPixel],vel_water_material_ratio,x_minus);
                            float fdash=(fx_plus-fx_minus)*inv_diff2;
                            float ffloatdash=(fx_plus-2.0*fx+fx_minus)*inv_diffsqr;
                            float xchange=fdash/(ffloatdash+1.0e-30);
                            //mexPrintf("Pixel %d Element %d f' %e f'' %e changeX: %e fx %e\n", iPixel, iElement, fdash,ffloatdash,xchange,fx);
                            if (fabs(xchange) < 1.0e-09)
                            {
                                lookup_times[iPixel*nelements+iElement]=fx/vel_material;
                                break;
                            }
                            else
                            {
                                interface_pointX=interface_pointX-xchange;
                            }

                        }

                    }
                }
            }
            break;

        case 2: // Immersed, direct,i.e. probe->frontwall->pixel (Fermat Solver) or vice versa
            if (direction_opt<1)
                {
                // Direction : probe->frontwall->pixel
                    if (travels_calculated<1)
                    {
                        //mexPrintf("calculating travel times -> probe to frontwall\n");
                        if (dimension_opt>2)
                        {
    #pragma omp for schedule(dynamic,CHUNKSIZE2)
                            for (iFront=0; iFront<nfrontwall;iFront++)
                            {
                                int fb_int=iFront*nelements;
                                for (iElement=0; iElement<nelements;iElement++)
                                {
                                    cur_time[fb_int+iElement]=travel_time_leg3d(probe_coordsX[iElement],probe_coordsY[iElement],probe_coordsZ[iElement],frontwall_coordsX[iFront],frontwall_coordsY[iFront],frontwall_coordsZ[iFront],inv_vel_water);
                                }
                            }
                        }
                        else
                        {
    #pragma omp for schedule(dynamic,CHUNKSIZE2)
                            for (iFront=0; iFront<nfrontwall;iFront++)
                            {
                                int fb_int=iFront*nelements;
                                for (iElement=0; iElement<nelements;iElement++)
                                {
                                    cur_time[fb_int+iElement]=travel_time_leg2d(probe_coordsX[iElement],probe_coordsZ[iElement],frontwall_coordsX[iFront],frontwall_coordsZ[iFront],inv_vel_water);
                                }
                            }
                        }
                    }
                }
            else
            {
                
            }
            
#pragma omp for schedule(dynamic,CHUNKSIZE_PIXEL)
            for (iPixel=0; iPixel<npixels;iPixel++)
            {
                
                for (iElement=0; iElement<nelements;iElement++)
                {
                    int iPixel2 = iPixel*nelements + iElement;
                    lookup_times[iPixel2]=10e10F;
                }
            }            
            if (dimension_opt>2)
            {
#pragma omp for schedule(dynamic,CHUNKSIZE)               
                for (iPixel=0; iPixel<npixels;iPixel++)
                {
                    for (iFront = 0; iFront<nfrontwall; iFront++)
                    {
                        int time_offset=iFront*nelements;
                        float fp_leg=travel_time_leg3d(pixel_coordsX[iPixel],pixel_coordsY[iPixel],pixel_coordsZ[iPixel],frontwall_coordsX[iFront],frontwall_coordsY[iFront],frontwall_coordsZ[iFront],inv_vel_material1);
                        for (iElement=0; iElement<nelements;iElement++)
                        {
                            int iPixel2 = iPixel*nelements + iElement;
                            float cur_travel_time = cur_time[time_offset+iElement] + fp_leg;
                            if (cur_travel_time < lookup_times[iPixel2])
                            {
                                lookup_times[iPixel2]=cur_travel_time;
                            }
                        }
                    }
                }                
            }
            else
            { 
#pragma omp for schedule(dynamic,CHUNKSIZE)               
                for (iPixel=0; iPixel<npixels;iPixel++)
                {
                    for (iFront = 0; iFront<nfrontwall; iFront++)
                    {
                        int time_offset=iFront*nelements;
                        float fp_leg=travel_time_leg2d(pixel_coordsX[iPixel],pixel_coordsZ[iPixel],frontwall_coordsX[iFront],frontwall_coordsZ[iFront],inv_vel_material1);
                        for (iElement=0; iElement<nelements;iElement++)
                        {
                            int iPixel2 = iPixel*nelements + iElement;
                            float cur_travel_time = cur_time[time_offset+iElement] + fp_leg;
                            if (cur_travel_time < lookup_times[iPixel2])
                            {
                                lookup_times[iPixel2]=cur_travel_time;
                            }
                        }
                    }
                }
            }

            break;
        case 3: //Immersed, skip path, i.e. probe->frontwall->backwall->pixel (Fermat Solver)
            if (direction_opt>0)
            {
                if (dimension_opt < 3)
                {
                    iFrontArray=(int*) malloc(nelements*sizeof(int));
                }
                cur_time2=(float*) malloc(nfrontwall*sizeof(float));
            }
            
            if (travels_calculated<1)
            {
                //Calculate distances/travel times to be used in minimum time search
                if (direction_opt<1)
                {
                    if (dimension_opt>2)
                    {
#pragma omp for schedule(dynamic,CHUNKSIZE2)                        
                        for (iBackwall=0; iBackwall<nbackwall;iBackwall++)
                        {
                            int fb_int=iBackwall*nfrontwall;
                            for (iFront=0; iFront<nfrontwall;iFront++)
                            {
                                travel_times_fb[fb_int+iFront]=travel_time_leg3d(frontwall_coordsX[iFront],frontwall_coordsY[iFront],frontwall_coordsZ[iFront],backwall_coordsX[iBackwall],backwall_coordsY[iBackwall],backwall_coordsZ[iBackwall],inv_vel_material1);
                            }
                        } 
#pragma omp for schedule(dynamic,CHUNKSIZE)
                        for (iElement=0; iElement<nelements;iElement++)
                        {
                            int fb_int=iElement*nfrontwall;
                            for (iFront=0; iFront<nfrontwall;iFront++)
                            {
                                travel_times_pf[fb_int+iFront]=travel_time_leg3d(probe_coordsX[iElement],probe_coordsY[iElement],probe_coordsZ[iElement],frontwall_coordsX[iFront],frontwall_coordsY[iFront],frontwall_coordsZ[iFront],inv_vel_water);
                            }
                        }
                    }
                    else
                    {
#pragma omp for schedule(dynamic,CHUNKSIZE2)                        
                        for (iBackwall=0; iBackwall<nbackwall;iBackwall++)
                        {
                            int fb_int=iBackwall*nfrontwall;
                            for (iFront=0; iFront<nfrontwall;iFront++)
                            {
                                travel_times_fb[fb_int+iFront]=travel_time_leg2d(frontwall_coordsX[iFront],frontwall_coordsZ[iFront],backwall_coordsX[iBackwall],backwall_coordsZ[iBackwall],inv_vel_material1);
                            }
                        }
#pragma omp for schedule(dynamic,CHUNKSIZE)
                        for (iElement=0; iElement<nelements;iElement++)
                        {
                            int fb_int=iElement*nfrontwall;
                            for (iFront=0; iFront<nfrontwall;iFront++)
                            {
                                travel_times_pf[fb_int+iFront]=travel_time_leg2d(probe_coordsX[iElement],probe_coordsZ[iElement],frontwall_coordsX[iFront],frontwall_coordsZ[iFront],inv_vel_water);
                            }
                        }
                    }

                    
                }
                else
                {
                    if (dimension_opt>2)
                    {
    #pragma omp for schedule(dynamic,CHUNKSIZE2)
                        for (iFront=0; iFront<nfrontwall;iFront++)
                        {
                            int fb_int=iFront*nbackwall;
                            for (iBackwall=0; iBackwall<nbackwall;iBackwall++)
                            {
                                travel_times_fb[fb_int+iBackwall]=travel_time_leg3d(frontwall_coordsX[iFront],frontwall_coordsY[iFront],frontwall_coordsZ[iFront],backwall_coordsX[iBackwall],backwall_coordsY[iBackwall],backwall_coordsZ[iBackwall],inv_vel_material1);
                            }
                        }
    #pragma omp for schedule(dynamic,CHUNKSIZE2)
                        for (iFront=0; iFront<nfrontwall;iFront++)
                        {
                            int fb_int=iFront*nelements;
                            for (iElement=0; iElement<nelements;iElement++)
                            {
                                travel_times_pf[fb_int+iElement]=travel_time_leg3d(probe_coordsX[iElement],probe_coordsY[iElement],probe_coordsZ[iElement],frontwall_coordsX[iFront],frontwall_coordsY[iFront],frontwall_coordsZ[iFront],inv_vel_water);
                            }
                        }                        
                    }
                    else
                    {
    #pragma omp for schedule(dynamic,CHUNKSIZE2)
                        for (iFront=0; iFront<nfrontwall;iFront++)
                        {
                            int fb_int=iFront*nbackwall;
                            for (iBackwall=0; iBackwall<nbackwall;iBackwall++)
                            {
                                travel_times_fb[fb_int+iBackwall]=travel_time_leg2d(frontwall_coordsX[iFront],frontwall_coordsZ[iFront],backwall_coordsX[iBackwall],backwall_coordsZ[iBackwall],inv_vel_material1);
                            }
                        }
    #pragma omp for schedule(dynamic,CHUNKSIZE2)
                        for (iFront=0; iFront<nfrontwall;iFront++)
                        {
                            int fb_int=iFront*nelements;
                            for (iElement=0; iElement<nelements;iElement++)
                            {
                                travel_times_pf[fb_int+iElement]=travel_time_leg2d(probe_coordsX[iElement],probe_coordsZ[iElement],frontwall_coordsX[iFront],frontwall_coordsZ[iFront],inv_vel_water);
                            }
                        }
                    }
                }
            }
            
            if (dimension_opt>2)
            {
                for (iPixel=0; iPixel<npixels;iPixel++)
                {
                    int fb_int=iPixel*nbackwall;
                    for (iBackwall=0; iBackwall<nbackwall;iBackwall++)
                    {
                        travel_times_bp[fb_int+iBackwall]=travel_time_leg3d(backwall_coordsX[iBackwall],backwall_coordsY[iBackwall],backwall_coordsZ[iBackwall],pixel_coordsX[iPixel],pixel_coordsY[iPixel],pixel_coordsZ[iPixel],inv_vel_material2);
                    }
                }                
            }
            else
            {
#pragma omp for schedule(dynamic,CHUNKSIZE)
                for (iPixel=0; iPixel<npixels;iPixel++)
                {
                    int fb_int=iPixel*nbackwall;
                    for (iBackwall=0; iBackwall<nbackwall;iBackwall++)
                    {
                        travel_times_bp[fb_int+iBackwall]=travel_time_leg2d(backwall_coordsX[iBackwall],backwall_coordsZ[iBackwall],pixel_coordsX[iPixel],pixel_coordsZ[iPixel],inv_vel_material2);
                    }
                }
            }
            
            
            if (direction_opt<1)
            {
                //Direction is probe->frontwall->backwall->pixels
                if (travels_calculated<1)
                {
#pragma omp for schedule(dynamic,CHUNKSIZE)
                    for (iElement=0; iElement<nelements;iElement++)
                    {
                        int time_offset=iElement*nbackwall;
                        int fb_int2=iElement*nfrontwall;
                        // Find best frontwall index for each backwall index
                        for (iBackwall=0; iBackwall<nbackwall;iBackwall++)
                        {
                            int fb_int = iBackwall*nfrontwall;
                            int time_offset2=time_offset+iBackwall;
                            cur_time[time_offset2]=10e10F;
                            for (iFront=0; iFront<nfrontwall;iFront++)
                            {


                                float cur_travel_time = travel_times_pf[fb_int2 + iFront] + travel_times_fb[fb_int + iFront];
                                if (cur_travel_time < cur_time[time_offset2])
                                {
                                    cur_time[time_offset2]=cur_travel_time;
                                }
                            }
                        }
                    }
                }
#pragma omp for schedule(dynamic,CHUNKSIZE)
                for (iElement=0; iElement<nelements;iElement++)
                {
                    int time_offset=iElement*nbackwall;
                    //Find best backwall index for each pixel
                    for (int iPixel=0; iPixel<npixels;iPixel++)
                    {
                        int iPixel2 = iPixel*nelements + iElement;
                        lookup_times[iPixel2]=10e10F;
                        int fb_int3 = iPixel*nbackwall;
                        for (iBackwall = 0; iBackwall<nbackwall; iBackwall++)
                        {
                            float cur_travel_time = cur_time[time_offset+iBackwall] + travel_times_bp[fb_int3+iBackwall];
                            if (cur_travel_time < lookup_times[iPixel2])
                            {
                                lookup_times[iPixel2]=cur_travel_time;
                            }
                        }
                    }
                }
            }
            else
            {
                //Direction is pixels->backwall->frontwall->probe
                if (dimension_opt<3)
                {
#pragma omp for schedule(dynamic,1)
                    for (iPixel=0; iPixel<npixels;iPixel++)
                    {
                        // Find best backwall index for each frontwall index
                        for (iFront=0; iFront<nfrontwall;iFront++)
                        {
                            int fb_int2=iPixel*nbackwall;
                            int fb_int = iFront*nbackwall;
                            //int iFront2=nbackwall+iFront;
                            cur_time2[iFront]=10e10F;
                            for (iBackwall=0; iBackwall<nbackwall;iBackwall++)
                            {
                                float cur_travel_time = travel_times_bp[fb_int2 + iBackwall] + travel_times_fb[fb_int + iBackwall];
                                if (cur_travel_time < cur_time2[iFront])
                                {
                                    cur_time2[iFront]=cur_travel_time;
                                }
                            }
                        }
                        //Find best frontwall index for each element
                        iFrontArray[0]=0;
                        iFrontArray[nelements-1]=nfrontwall-1;
                        for (int iSearch=0; iSearch<nelements;iSearch++)
                        {
                            iElement=searchOrder[iSearch];
                            int iPixel2 = iPixel*nelements + iElement;

                            int iPar1=iParent1[iSearch];
                            int iPar2=iParent2[iSearch];
                            int iFrontStart=iFrontArray[iPar1];
                            int iFrontEnd=iFrontArray[iPar2]+1;
                            lookup_times[iPixel2]=10e10F;
                            for (iFront = iFrontStart; iFront<iFrontEnd; iFront++)
                            {
                                //int iFront2=nbackwall+iFront;
                                int fb_int3 = iFront*nelements + iElement;
                                float cur_travel_time = cur_time2[iFront] + travel_times_pf[fb_int3];
                                if (cur_travel_time < lookup_times[iPixel2])
                                {
                                    lookup_times[iPixel2]=cur_travel_time;
                                    iFrontArray[iElement]=iFront;
                                }
                            }
                        }
                    }
                }
                else
                {
                    #pragma omp for schedule(dynamic,1)
                    for (iPixel=0; iPixel<npixels;iPixel++)
                    {
                        // Find best backwall index for each frontwall index
                        for (iFront=0; iFront<nfrontwall;iFront++)
                        {
                            int fb_int2=iPixel*nbackwall;
                            int fb_int = iFront*nbackwall;
                            //int iFront2=nbackwall+iFront;
                            cur_time2[iFront]=10e10F;
                            for (iBackwall=0; iBackwall<nbackwall;iBackwall++)
                            {
                                float cur_travel_time = travel_times_bp[fb_int2 + iBackwall] + travel_times_fb[fb_int + iBackwall];
                                if (cur_travel_time < cur_time2[iFront])
                                {
                                    cur_time2[iFront]=cur_travel_time;
                                }
                            }
                        }
                        //Find best frontwall index for each element
                        iFrontArray[0]=0;
                        iFrontArray[nelements-1]=nfrontwall-1;
                        for (int iElement=0; iElement<nelements;iElement++)
                        {
                            int iPixel2 = iPixel*nelements + iElement;
                            lookup_times[iPixel2]=10e10F;
                            for (iFront = 0; iFront<nfrontwall; iFront++)
                            {
                                //int iFront2=nbackwall+iFront;
                                int fb_int3 = iFront*nelements + iElement;
                                float cur_travel_time = cur_time2[iFront] + travel_times_pf[fb_int3];
                                if (cur_travel_time < lookup_times[iPixel2])
                                {
                                    lookup_times[iPixel2]=cur_travel_time;
                                }
                            }
                        }
                    }
                }
            }
            
            //Free parallel pointers linked to this path_opt
            if (direction_opt>0)
            {
                if (dimension_opt<3)
                {
                    free(iFrontArray);
                }
                free(cur_time2);
            }
            
            break;
    }
}


// Free up temporary (non-parallel) pointers/arrays

if (path_opt>2)
{
    // Backwall to pixel (or vice versa)
    free(travel_times_bp);
    if (travels_calculated<1)
    {
        free(travel_times_fb);
        free(travel_times_pf);   
    }
    if (travels_calculated<0)
    {
        free(cur_time);   
    }

}
else if (path_opt>1)
{
    // Frontwall to pixel (or vice versa)
    if (travels_calculated<0)
    {
        free(cur_time);   
    }
}

// End of gateway function

}
