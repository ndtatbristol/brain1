function [LR_offset,LR_angle]=fn_linear_line_best_fit(x,y,err_tolerance)

    %x=exp_data.array.el_xc.';
    xlength=length(x);
    %y=(time_frontwall_PE-instrument_delay)*scaling;
    X = [ones(length(x),1) x];
    b = X\y;
    for ii=1:xlength
        LR_offset=b(1);
        LR_angle=asin(b(2))*180/pi;
        y_calc=LR_offset+b(2)*x;  
        y_calc_exceeded=find(abs(y_calc-y) > err_tolerance);
        try
            dummy=y_calc_exceeded(1);
        catch
            %disp(['No values exceeding error threshold in loop: ', int2str(ii)])
            break;
        end
        y_mask=ones(size(y));
        [~,qq2]=max(abs(y_calc-y));
        y_mask(qq2)=0;
        reduced=find(y_mask>0);
        y_reduced=y(reduced);
        y=y_reduced;
        x_reduced=x(reduced);
        x=x_reduced;
        X = [ones(length(x),1) x];
        b = X\y;
    end
    
end