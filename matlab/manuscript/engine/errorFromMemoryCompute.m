function f = errorFromMemoryCompute(slopeJControllerVsContext,pInputNow_ConsideredGood,paramFixed,contextGaitNow)
% Compute the error between the stored policy and the current controller.

pInputMemoryNow = paramFixed.storedmemory.nominalControl + ...
        slopeJControllerVsContext*(contextGaitNow-paramFixed.storedmemory.nominalContext);

f = (pInputNow_ConsideredGood-pInputMemoryNow)'*(pInputNow_ConsideredGood-pInputMemoryNow);

end
