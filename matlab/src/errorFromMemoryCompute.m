function f = errorFromMemoryCompute(slopeJControllerVsContext,pInputNow_ConsideredGood,paramFixed,contextGaitNow)
% Return the squared distance between a controller and the memory's prediction.
% Inputs:  slopeJControllerVsContext - memory slopes, controller x context
%          pInputNow_ConsideredGood  - 8-D controller
%          paramFixed                - uses storedmemory.nominalControl and nominalContext
%          contextGaitNow            - context, the two belt speeds
% Outputs: f - squared Euclidean prediction error

pInputMemoryNow = paramFixed.storedmemory.nominalControl + ...
        slopeJControllerVsContext*(contextGaitNow-paramFixed.storedmemory.nominalContext);

f = (pInputNow_ConsideredGood-pInputMemoryNow)'*(pInputNow_ConsideredGood-pInputMemoryNow);

end
