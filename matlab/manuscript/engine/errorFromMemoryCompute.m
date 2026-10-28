function f = errorFromMemoryCompute(slopeJControllerVsContext,pInputNow_ConsideredGood,paramFixed,contextGaitNow)
% Return the squared distance between a controller and the stored-memory prediction.
% Inputs:  slopeJControllerVsContext - change in controller per unit change in context
%          pInputNow_ConsideredGood - current controller
%          paramFixed.storedmemory - nominalControl and nominalContext of the memory
%          contextGaitNow - current context (belt speeds)
% Outputs: f - squared prediction error

pInputMemoryNow = paramFixed.storedmemory.nominalControl + ...
        slopeJControllerVsContext*(contextGaitNow-paramFixed.storedmemory.nominalContext);

f = (pInputNow_ConsideredGood-pInputMemoryNow)'*(pInputNow_ConsideredGood-pInputMemoryNow);

end
