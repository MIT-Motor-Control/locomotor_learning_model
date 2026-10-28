function [yKneeList,zKneeList] = kneeGivenBodyFoot(yBodyList,zBodyList,yFootList,zFootList)
% Construct swing-leg knee coordinates for gait visualization.
% Inputs:  yBodyList, zBodyList - body positions; yFootList, zFootList - swing-foot positions
% Outputs: yKneeList, zKneeList - knee positions

yKneeList = zeros(size(yBodyList)); zKneeList = yKneeList;

for jFrame = 1:length(yBodyList)
    [yKneeList(jFrame),zKneeList(jFrame)] = ...
        kneeGivenBodyFootScalar(yBodyList(jFrame),zBodyList(jFrame),yFootList(jFrame),zFootList(jFrame));
end

end

function [yKnee,zKnee] = kneeGivenBodyFootScalar(yBody,zBody,yFoot,zFoot)
% Place the knee by rotating the body-to-midpoint vector by acos(body-to-foot distance).

yMid = (yBody+yFoot)/2;
zMid = (zBody+zFoot)/2;

legLengthNow = norm([yBody-yFoot; zBody-zFoot]);
legLengthNominal = 1;

alpha = acos(legLengthNow/legLengthNominal);
alpha = abs(alpha);

R = [cos(alpha) -sin(alpha); sin(alpha) cos(alpha)];

temp = [yBody; zBody] + R*[yMid-yBody; zMid-zBody];
yKnee = temp(1); zKnee = temp(2);

end
