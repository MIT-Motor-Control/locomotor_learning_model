function beltSpeedsImposed = makeTreadmillSpeed_Split(paramFixed)
% Build a split-belt speed profile with linear transition ramps and plot it in figure 2555.
% Inputs:  paramFixed.speedProtocol - protocol name; paramFixed.transitionTime - ramp duration (s)
% Outputs: beltSpeedsImposed - tList (model time), footSpeed1List and footSpeed2List
%          (negative values move backward), footAcc1List and footAcc2List (constant per segment)

L = 0.95; g = 9.81;
timeScaling = sqrt(L/g);

delta = 0.0328;
vNormal = -0.3276;
vFast = vNormal - 5*delta;
vSlow = vNormal + 5*delta;

switch paramFixed.speedProtocol
    case 'single speed'

        tDurationTransition = paramFixed.transitionTime/timeScaling;

        tDuration1 = 9*60/timeScaling;
        footSpeed1_phase1 = vNormal;
        footSpeed2_phase1 = vNormal;

        tDuration2 = 1*60/timeScaling;
        footSpeed1_phase2 = vNormal;
        footSpeed2_phase2 = vNormal;

        tDuration3 = 1*60/timeScaling;
        footSpeed1_phase3 = vNormal;
        footSpeed2_phase3 = vNormal;

        tDuration4 = 1*60/timeScaling;
        footSpeed1_phase4 = vNormal;
        footSpeed2_phase4 = vNormal;

        tStore = [0; tDuration1; tDuration2; tDuration3; tDuration4];
        tStore = cumsum(tStore);

        footSpeed1Store = [footSpeed1_phase1; footSpeed1_phase1; ...
            footSpeed1_phase2; footSpeed1_phase3; footSpeed1_phase4];
        footSpeed2Store = [footSpeed2_phase1; footSpeed2_phase1; ...
            footSpeed2_phase2; footSpeed2_phase3; footSpeed2_phase4];

    case 'classic split belt'

        tDurationTransition = paramFixed.transitionTime/timeScaling;

        tDuration1 = 1*60/timeScaling;
        footSpeed1_phase1 = vNormal;
        footSpeed2_phase1 = vNormal;

        tDuration2 = 5*60/timeScaling;
        footSpeed1_phase2 = vNormal;
        footSpeed2_phase2 = vNormal;

        tDuration3 = 45*60/timeScaling;
        footSpeed1_phase3 = vFast;
        footSpeed2_phase3 = vSlow;

        tDuration4 = 5*60/timeScaling;
        footSpeed1_phase4 = vNormal;
        footSpeed2_phase4 = vNormal;

        tStore = [0; tDuration1; tDuration2; tDuration3; tDuration4];
        tStore = cumsum(tStore);

        footSpeed1Store = [footSpeed1_phase1; footSpeed1_phase1; ...
            footSpeed1_phase2; footSpeed1_phase3; footSpeed1_phase4];
        footSpeed2Store = [footSpeed2_phase1; footSpeed2_phase1; ...
            footSpeed2_phase2; footSpeed2_phase3; footSpeed2_phase4];

end

tStore_new = 0;
footSpeed1Store_new = footSpeed1Store(1);
footSpeed2Store_new = footSpeed2Store(1);

for iTran = 2:length(tStore)
    if iTran<length(tStore)
        tStore_new = [tStore_new; tStore(iTran); ...
            tStore(iTran)+tDurationTransition];
        footSpeed1Store_new = [footSpeed1Store_new; footSpeed1Store(iTran); footSpeed1Store(iTran+1)];
        footSpeed2Store_new = [footSpeed2Store_new; footSpeed2Store(iTran); footSpeed2Store(iTran+1)];
    else
        tStore_new = [tStore_new; tStore(iTran)];
        footSpeed1Store_new = [footSpeed1Store_new; footSpeed1Store(iTran)];
        footSpeed2Store_new = [footSpeed2Store_new; footSpeed2Store(iTran)];
    end
end

figure(2555);
plot(tStore_new,abs(footSpeed1Store_new),'linewidth',2); hold on;
plot(tStore_new,abs(footSpeed2Store_new),'linewidth',2);
xlabel('t'); ylabel('treadmill speeds (non-dimensional)');
legend('(abs) fast belt','(abs) slow belt');
ylim([0 abs(vFast)*1.25]);
title('Split belt speed change protocol');

beltSpeedsImposed.tList = tStore_new;
beltSpeedsImposed.footSpeed1List = footSpeed1Store_new;
beltSpeedsImposed.footSpeed2List = footSpeed2Store_new;

a1List = ...
    diff(beltSpeedsImposed.footSpeed1List)./diff(beltSpeedsImposed.tList);
a2List = ...
    diff(beltSpeedsImposed.footSpeed2List)./diff(beltSpeedsImposed.tList);
a1List = [a1List; 0]; a2List = [a2List; 0];

beltSpeedsImposed.footAcc1List = a1List;
beltSpeedsImposed.footAcc2List = a2List;

end
