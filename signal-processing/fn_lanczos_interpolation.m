function [val]=fn_lanczos_interpolation( x, t, n,a)

    % Uses Lanzos Interpolation (a = kernel size)
    % x = array containing variable to interpolate
    % t = position in x to interpolate value for, e.g. 3.644459 lies between entry 3 and 4 in x. 
    % n = number of values in x (should exceed t+a, otherwise interpolation is skipped)

    % see https://en.wikipedia.org/wiki/Lanczos_resampling for further details
    
    i_min=floor(t - a + 1);
    i_max=floor(t + a);
    val=0;
    if (i_min < 1 || i_max >= n)
        %val= 0.0;
    else
        for i=i_min:i_max
            val=val+x(i)*lanczos_window2(t-i,a);
        end
    end

    function [val]=lanczos_window2(t,a)
        if (abs(t) < 1e-7)   
            val=1;
        else
            pix=pi*t;
            val= a*sin(pix)*sin(pix/a)/(pix*pix); %*sinc_normalised(t)*sinc_normalised(t/a);
        end
    end

end
