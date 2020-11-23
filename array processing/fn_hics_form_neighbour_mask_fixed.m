function fraction_mask = fn_hics_form_neighbour_mask_fixed(q_array,view_psf,psf_safety_factor,pixel_size)

fraction_mask=zeros(size(q_array));
[nz,nx]=size(q_array);

for iz=1:nz
    for ix=1:nx
        if (q_array(iz,ix) ~=1)
            continue
        end
        ip=round(psf_safety_factor*view_psf/pixel_size);
        iCount=0; iCount2=0;
        for izz=-ip:1:ip
            if (izz+iz < 1 || izz+iz > nz)
                continue
            end
            for ixx=-ip:1:ip
                if (ixx+ix < 1 || ixx+ix > nx)
                    continue
                end
                %if (q_array(iz,ix) < 2)
                iCount=iCount+1;
                if (q_array(izz+iz,ixx+ix) > 0)
                    iCount2=iCount2+1;
                end
                %end
            end
        end
        fraction_mask(iz,ix)=1.0*iCount2/iCount;
    end
end

end