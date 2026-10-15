function [fall_info, y_pos] = checkModelFall(tlist, statelist, leglength, iStride)
    % Initialize fall flag
    fall_info = [0 0];

    % Initialize arrays to store point mass positions
    x_positions = zeros(length(tlist), 1);
    y_positions = zeros(length(tlist), 1);

    % Calculate the positions of the point mass
    for i = 1:length(tlist)
        angleTheta = statelist(i, 1); % Assuming angleTheta is the first state variable
        x_positions(i) = leglength * sin(angleTheta);
        y_positions(i) = leglength * cos(angleTheta);

        % Check for weird circles or negative positions
        % if angleTheta < -pi/2 || angleTheta > pi/2 || y_positions(i) < 0
        if  y_positions(i) < 0
            fall_info(1) = 1;
        end
    end
    y_pos = min(y_positions);

    % Record the stride at which the fall occurred. This assignment existed in the
    % code version that generated the original bundled data (fall_flag(2) == stride
    % index == numel(time)); it was lost when the display block below was commented
    % out, leaving fall_flag(2) = 0 in regenerated data.
    if fall_info(1) == 1
        fall_info(2) = iStride;
    end

    % Display fall flag status
    % if fall_info(1) == 1
    %     disp(['Model fell at stride: ', num2str(iStride)]);
    %     fall_info(2) = iStride;
    % 
    %     figure;
    %     axis equal;
    %     hold on;
    %     ylabel('Vertical Height');
    %     title('Model Stance Phase');
    % 
    %     plot(x_positions, y_positions, '.-', 'Color','b');
    %     plot([0, x_positions(end)], [0, y_positions(end)], 'ro-');
    %     plot(x_positions(end), y_positions(end), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    % 
    % end   
end