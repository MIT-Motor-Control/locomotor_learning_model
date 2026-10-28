function g = gradientOfErrorFromMemoryCompute(slopeJControllerVsContext,pInputNow_ConsideredGood,paramFixed,contextGaitNow)
% Return the forward-difference gradient of errorFromMemoryCompute with respect to the slopes.
% Inputs:  as for errorFromMemoryCompute
% Outputs: g - gradient, same size as slopeJControllerVsContext

f0 = errorFromMemoryCompute(slopeJControllerVsContext,pInputNow_ConsideredGood,paramFixed,contextGaitNow);

h = 1e-4;

g = [];

for iCount = 1:size(slopeJControllerVsContext,1)
    for jCount = 1:size(slopeJControllerVsContext,2)

        slopeJControllerVsContextNow = slopeJControllerVsContext;
        slopeJControllerVsContextNow(iCount,jCount) = ...
            slopeJControllerVsContextNow(iCount,jCount) + h;

        f = errorFromMemoryCompute(slopeJControllerVsContextNow,pInputNow_ConsideredGood,paramFixed,contextGaitNow);

        g(iCount,jCount) = (f-f0)/h;

    end
end

end
