function [focal_laws]=fn_determine_focal_laws(combine_opt,nview1,nview2,instrument_delay,solver_opt,dimension_opt,direction_opt,exp_data, vel_water,vel_long,vel_shear,probe_coordsX,probe_coordsY,probe_coordsZ,pixel_coordsX,pixel_coordsY,pixel_coordsZ,frontwall_coordX,frontwall_coordY,frontwall_coordZ,backwall_coordX,backwall_coordY,backwall_coordZ,sample_freq,travels_calculated,varargin)

%Different Ray Paths Allowed
%1  -   Contact
%2  -   Immersion Direct: (Water L +) Material L
%3  -   Immersion Direct: (Water L +) Material T
%4  -   Immersion Skip: (Water L +) Material L + Material L (via Backwall)
%5  -   Immersion Skip: (Water L +) Material L + Material T (via Backwall)
%6  -   Immersion Skip: (Water L +) Material T + Material L (via Backwall)
%7  -   Immersion Skip: (Water L +) Material T + Material T (via Backwall)

if (length(varargin)>0)
    precision=varargin{1};
else
    precision='single';
end

rayPathsRequired=zeros(7,1);
rayPathsRequired2=zeros(2,nview2-nview1+1);


focal_laws.views_start=nview1;
focal_laws.views_end=nview2;

iview2=0;
for iview=nview1:nview2
    cur_view=iview-1;
    iview2=iview2+1;
    switch (cur_view)
        case -2 %2nd frontwall reflection
            view_name='FW_2nd';    
            rayPathsRequired2(1,iview2)=2; rayPathsRequired2(2,iview2)=2;
        case -1 %contact
            view_name='Contact';    
            rayPathsRequired2(1,iview2)=1; rayPathsRequired2(2,iview2)=1;
        case 0 %L-L
            view_name='L-L';    
            rayPathsRequired2(1,iview2)=2; rayPathsRequired2(2,iview2)=2;
        case 1 %L-T
            view_name='L-T';    
            rayPathsRequired2(1,iview2)=2; rayPathsRequired2(2,iview2)=3;
        case 2 %T-T
            view_name='T-T';    
            rayPathsRequired2(1,iview2)=3; rayPathsRequired2(2,iview2)=3;
        case 3 %LL-L
            view_name='LL-L';    
            rayPathsRequired2(1,iview2)=4; rayPathsRequired2(2,iview2)=2;
        case 4 %LL-T
            view_name='LL-T';    
            rayPathsRequired2(1,iview2)=4; rayPathsRequired2(2,iview2)=3;
        case 5 %LT-L
            view_name='LT-L';    
            rayPathsRequired2(1,iview2)=5; rayPathsRequired2(2,iview2)=2;    
        case 6 %LT-T
            view_name='LT-T';    
            rayPathsRequired2(1,iview2)=5; rayPathsRequired2(2,iview2)=3;  
        case 7 %TL-L
            view_name='TL-L';    
            rayPathsRequired2(1,iview2)=6; rayPathsRequired2(2,iview2)=2;             
        case 8 %TL-T
            view_name='TL-T';    
            rayPathsRequired2(1,iview2)=6; rayPathsRequired2(2,iview2)=3;             
        case 9 %TT-L
            view_name='TT-L';    
            rayPathsRequired2(1,iview2)=7; rayPathsRequired2(2,iview2)=2;             
        case 10 %TT-T
            view_name='TT-T';     
            rayPathsRequired2(1,iview2)=7; rayPathsRequired2(2,iview2)=3;            
        case 11 %LL-LL
            view_name='LL-LL';     
            rayPathsRequired2(1,iview2)=4; rayPathsRequired2(2,iview2)=4;             
        case 12 %LL-LT
            view_name='LL-LT';     
            rayPathsRequired2(1,iview2)=4; rayPathsRequired2(2,iview2)=6;              
        case 13 %LL-TL
            view_name='LL-TL';            
            rayPathsRequired2(1,iview2)=4; rayPathsRequired2(2,iview2)=5;
        case 14 %LL-TT
            view_name='LL-TT';
            rayPathsRequired2(1,iview2)=4; rayPathsRequired2(2,iview2)=7;            
        case 15 %LT-LT
            view_name='LT-LT';
            rayPathsRequired2(1,iview2)=5; rayPathsRequired2(2,iview2)=6; 
        case 16 %LT-TL
            view_name='LT-TL';
            rayPathsRequired2(1,iview2)=5; rayPathsRequired2(2,iview2)=5;              
        case 17 %LT-TT
            view_name='LT-TT';
            rayPathsRequired2(1,iview2)=5; rayPathsRequired2(2,iview2)=7;              
        case 18 %TL-LT
            view_name='TL-LT';
            rayPathsRequired2(1,iview2)=6; rayPathsRequired2(2,iview2)=6;              
        case 19 %TL-TT
            view_name='TL-TT';
            rayPathsRequired2(1,iview2)=6; rayPathsRequired2(2,iview2)=7;   
        case 20 %TT-TT
            view_name='TT-TT';
            rayPathsRequired2(1,iview2)=7; rayPathsRequired2(2,iview2)=7;             
        otherwise
            disp('Unknown view specified')
            return
    end
    
    focal_laws.name{iview2}=view_name;
end

nviews=iview2;

%Need to compute required ray paths only
for iview=1:nviews
    rayPathsRequired(rayPathsRequired2(1:2,iview))=1;
end

rayPathsRequired3=find(rayPathsRequired == 1);
nRayPaths=length(rayPathsRequired3);
rayPathsRequired4=zeros(length(rayPathsRequired),1);
nelements=length(exp_data.array.el_xc);
npixels=length(pixel_coordsX);


switch (precision)
    case 'single'
        focal_laws.path=single(zeros(nelements,npixels,nRayPaths));
        %Ensure all non-integers are floats prior to calling mex function
        probe_coordsX=single(probe_coordsX);
        probe_coordsY=single(probe_coordsY);
        probe_coordsZ=single(probe_coordsZ);
        pixel_coordsX=single(pixel_coordsX);
        pixel_coordsY=single(pixel_coordsY);
        pixel_coordsZ=single(pixel_coordsZ);
        frontwall_coordX=single(frontwall_coordX);
        frontwall_coordY=single(frontwall_coordY);
        frontwall_coordZ=single(frontwall_coordZ);
        backwall_coordX=single(backwall_coordX);
        backwall_coordY=single(backwall_coordY);
        backwall_coordZ=single(backwall_coordZ);
        vel_water=single(vel_water);
        sample_freq=single(sample_freq);
        for i=1:nRayPaths
            ii=rayPathsRequired3(i);
            rayPathsRequired4(ii)=i;
            switch (ii)
                case 1 %Contact
                    if (dimension_opt>2)
                        lookup_time=mex_lookup_times8SP(int32(0),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsY,probe_coordsZ,pixel_coordsX,pixel_coordsY,pixel_coordsZ,vel_water,0.0,0.0,1.0/(exp_data.time(2)-exp_data.time(1)),int32(travels_calculated));
                    else
                        lookup_time=mex_lookup_times8SP(int32(0),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,vel_water,single(0.0),single(0.0),sample_freq,int32(travels_calculated));
                    end
                case 2 %L
                    vel_material_tx1=single(vel_long);
                    vel_material_tx2=single(0.0);
                    if (dimension_opt>2)
                        lookup_time=mex_lookup_times8SP(int32(2),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsY,probe_coordsZ,pixel_coordsX,pixel_coordsY,pixel_coordsZ,frontwall_coordX,frontwall_coordY,frontwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                    else
                        if (solver_opt == 1)
                            lookup_time=mex_lookup_times8SP(int32(1),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        else
                            lookup_time=mex_lookup_times8SP(int32(2),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        end
                    end
                case 3 %T
                    vel_material_tx1=single(vel_shear);
                    vel_material_tx2=single(0.0);
                    if (dimension_opt>2)
                        lookup_time=mex_lookup_times8SP(int32(2),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsY,probe_coordsZ,pixel_coordsX,pixel_coordsY,pixel_coordsZ,frontwall_coordX,frontwall_coordY,frontwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                    else
                        if (solver_opt == 1)
                            lookup_time=mex_lookup_times8SP(int32(1),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        else
                            lookup_time=mex_lookup_times8SP(int32(2),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        end
                    end            
                case 4 %LL
                    vel_material_tx1=single(vel_long);
                    vel_material_tx2=single(vel_long);
                    if (dimension_opt>2)
                        lookup_time=mex_lookup_times8SP(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsY,probe_coordsZ,pixel_coordsX,pixel_coordsY,pixel_coordsZ,frontwall_coordX,frontwall_coordY,frontwall_coordZ,backwall_coordX,backwall_coordY,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                    else
                        if (solver_opt == 1)
                            lookup_time=mex_lookup_times8SP(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,backwall_coordX,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        else
                            lookup_time=mex_lookup_times8SP(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,backwall_coordX,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        end
                    end  
                case 5 %LT
                    vel_material_tx1=single(vel_long);
                    vel_material_tx2=single(vel_shear);
                    if (dimension_opt>2)
                        lookup_time=mex_lookup_times8SP(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsY,probe_coordsZ,pixel_coordsX,pixel_coordsY,pixel_coordsZ,frontwall_coordX,frontwall_coordY,frontwall_coordZ,backwall_coordX,backwall_coordY,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                    else
                        if (solver_opt == 1)
                            lookup_time=mex_lookup_times8SP(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,backwall_coordX,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        else
                            lookup_time=mex_lookup_times8SP(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,backwall_coordX,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        end
                    end              
                case 6 %TL
                    vel_material_tx1=single(vel_shear);
                    vel_material_tx2=single(vel_long);
                    if (dimension_opt>2)
                        lookup_time=mex_lookup_times8SP(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsY,probe_coordsZ,pixel_coordsX,pixel_coordsY,pixel_coordsZ,frontwall_coordX,frontwall_coordY,frontwall_coordZ,backwall_coordX,backwall_coordY,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                    else
                        if (solver_opt == 1)
                            lookup_time=mex_lookup_times8SP(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,backwall_coordX,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        else
                            lookup_time=mex_lookup_times8SP(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,backwall_coordX,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        end
                    end            
                case 7 %TT
                    vel_material_tx1=single(vel_shear);
                    vel_material_tx2=single(vel_shear);
                    if (dimension_opt>2)
                        lookup_time=mex_lookup_times8SP(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsY,probe_coordsZ,pixel_coordsX,pixel_coordsY,pixel_coordsZ,frontwall_coordX,frontwall_coordY,frontwall_coordZ,backwall_coordX,backwall_coordY,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                    else
                        if (solver_opt == 1)
                            lookup_time=mex_lookup_times8SP(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,backwall_coordX,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        else
                            lookup_time=mex_lookup_times8SP(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,backwall_coordX,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        end
                    end
                otherwise
                    disp('Unknown ray path type')
                    return
            end

            %Include 50/50 split in instrument delay into path information (delay will then be shared equally between tx/rx components)
            focal_laws.path(:,:,i)=lookup_time+single(0.5*instrument_delay);
        end
    otherwise
        focal_laws.path=zeros(nelements,npixels,nRayPaths);
        for i=1:nRayPaths
            ii=rayPathsRequired3(i);
            rayPathsRequired4(ii)=i;
            switch (ii)
                case 1 %Contact
                    if (dimension_opt>2)
                        lookup_time=mex_lookup_times8(int32(0),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsY,probe_coordsZ,pixel_coordsX,pixel_coordsY,pixel_coordsZ,vel_water,0.0,0.0,1.0/(exp_data.time(2)-exp_data.time(1)),int32(travels_calculated));
                    else
                        lookup_time=mex_lookup_times8(int32(0),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,vel_water,0.0,0.0,1.0/(exp_data.time(2)-exp_data.time(1)),int32(travels_calculated));
                    end
                case 2 %L
                    vel_material_tx1=vel_long;
                    vel_material_tx2=0.0;
                    if (dimension_opt>2)
                        lookup_time=mex_lookup_times8(int32(2),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsY,probe_coordsZ,pixel_coordsX,pixel_coordsY,pixel_coordsZ,frontwall_coordX,frontwall_coordY,frontwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                    else
                        if (solver_opt == 1)
                            lookup_time=mex_lookup_times8(int32(1),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        else
                            lookup_time=mex_lookup_times8(int32(2),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        end
                    end
                case 3 %T
                    vel_material_tx1=vel_shear;
                    vel_material_tx2=0.0;
                    if (dimension_opt>2)
                        lookup_time=mex_lookup_times8(int32(2),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsY,probe_coordsZ,pixel_coordsX,pixel_coordsY,pixel_coordsZ,frontwall_coordX,frontwall_coordY,frontwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                    else
                        if (solver_opt == 1)
                            lookup_time=mex_lookup_times8(int32(1),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        else
                            lookup_time=mex_lookup_times8(int32(2),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        end
                    end            
                case 4 %LL
                    vel_material_tx1=vel_long;
                    vel_material_tx2=vel_long;
                    if (dimension_opt>2)
                        lookup_time=mex_lookup_times8(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsY,probe_coordsZ,pixel_coordsX,pixel_coordsY,pixel_coordsZ,frontwall_coordX,frontwall_coordY,frontwall_coordZ,backwall_coordX,backwall_coordY,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                    else
                        if (solver_opt == 1)
                            lookup_time=mex_lookup_times8(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,backwall_coordX,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        else
                            lookup_time=mex_lookup_times8(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,backwall_coordX,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        end
                    end  
                case 5 %LT
                    vel_material_tx1=vel_long;
                    vel_material_tx2=vel_shear;
                    if (dimension_opt>2)
                        lookup_time=mex_lookup_times8(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsY,probe_coordsZ,pixel_coordsX,pixel_coordsY,pixel_coordsZ,frontwall_coordX,frontwall_coordY,frontwall_coordZ,backwall_coordX,backwall_coordY,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                    else
                        if (solver_opt == 1)
                            lookup_time=mex_lookup_times8(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,backwall_coordX,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        else
                            lookup_time=mex_lookup_times8(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,backwall_coordX,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        end
                    end              
                case 6 %TL
                    vel_material_tx1=vel_shear;
                    vel_material_tx2=vel_long;
                    if (dimension_opt>2)
                        lookup_time=mex_lookup_times8(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsY,probe_coordsZ,pixel_coordsX,pixel_coordsY,pixel_coordsZ,frontwall_coordX,frontwall_coordY,frontwall_coordZ,backwall_coordX,backwall_coordY,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                    else
                        if (solver_opt == 1)
                            lookup_time=mex_lookup_times8(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,backwall_coordX,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        else
                            lookup_time=mex_lookup_times8(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,backwall_coordX,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        end
                    end            
                case 7 %TT
                    vel_material_tx1=vel_shear;
                    vel_material_tx2=vel_shear;
                    if (dimension_opt>2)
                        lookup_time=mex_lookup_times8(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsY,probe_coordsZ,pixel_coordsX,pixel_coordsY,pixel_coordsZ,frontwall_coordX,frontwall_coordY,frontwall_coordZ,backwall_coordX,backwall_coordY,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                    else
                        if (solver_opt == 1)
                            lookup_time=mex_lookup_times8(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,backwall_coordX,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        else
                            lookup_time=mex_lookup_times8(int32(3),int32(dimension_opt),int32(direction_opt),probe_coordsX,probe_coordsZ,pixel_coordsX,pixel_coordsZ,frontwall_coordX,frontwall_coordZ,backwall_coordX,backwall_coordZ,vel_water,vel_material_tx1,vel_material_tx2,sample_freq,int32(travels_calculated));
                        end
                    end
                otherwise
                    disp('Unknown ray path type')
                    return
            end

            %Include 50/50 split in instrument delay into path information (delay will then be shared equally between tx/rx components)
            focal_laws.path(:,:,i)=lookup_time+0.5*instrument_delay;
        end
end

focal_laws.raypaths=rayPathsRequired4(rayPathsRequired2);
focal_laws.count=nviews;

%combine_opt=1;
if (combine_opt)
    %% Combine all focal laws into 1 big focal law
    focal_laws_old=focal_laws;
    focal_laws=[];
    focal_laws.count=1;
    focal_laws.combined=1;
    focal_laws.name=focal_laws_old.name;
    focal_laws.views=nviews;
    focal_laws.pixels=npixels;
    focal_laws.raypaths=[1;2];
    focal_laws.views_start=nview1;
    focal_laws.views_end=nview2;
    switch (precision)
        case 'single'
            focal_laws.path_tx=single(zeros(nelements,npixels*nviews));
            focal_laws.path_rx=single(zeros(nelements,npixels*nviews));
        otherwise
            focal_laws.path_tx=zeros(nelements,npixels*nviews);
            focal_laws.path_rx=zeros(nelements,npixels*nviews);
    end
    for i=1:nviews
        tx_path=focal_laws_old.raypaths(1,i); rx_path=focal_laws_old.raypaths(2,i);
        focal_laws.path_tx(:,(i-1)*npixels+1:i*npixels)=focal_laws_old.path(:,:,tx_path);
        focal_laws.path_rx(:,(i-1)*npixels+1:i*npixels)=focal_laws_old.path(:,:,rx_path);
    end
    focal_laws.path_tx=focal_laws.path_tx.';
    focal_laws.path_rx=focal_laws.path_rx.';
end
    


end
            
            