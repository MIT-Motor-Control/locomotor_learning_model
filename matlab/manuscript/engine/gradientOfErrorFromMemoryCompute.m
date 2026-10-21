function g = gradientOfErrorFromMemoryCompute(slopeJControllerVsContext,pInputNow_ConsideredGood,paramFixed,contextGaitNow)
% Differentiate the stored-policy prediction error with respect to its slopes.

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
