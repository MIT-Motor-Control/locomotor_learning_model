function [fall_info, y_pos] = checkModelFall(tlist, statelist, leglength, iStride)
% Detect a fall when the point mass passes below ground level.
    fall_info = [0 0];

    x_positions = zeros(length(tlist), 1);
    y_positions = zeros(length(tlist), 1);

    for i = 1:length(tlist)
        angleTheta = statelist(i, 1);
        x_positions(i) = leglength * sin(angleTheta);
        y_positions(i) = leglength * cos(angleTheta);

        if  y_positions(i) < 0
            fall_info(1) = 1;
        end
    end
    y_pos = min(y_positions);

    % Record the stride of the fall.
    if fall_info(1) == 1
        fall_info(2) = iStride;
    end

end
