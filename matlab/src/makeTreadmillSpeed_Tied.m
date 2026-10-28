function beltSpeedsImposed = makeTreadmillSpeed_Tied(paramFixed)
% Build a tied-belt speed profile with linear transition ramps.
% Not called by default; to use it, call it in loadProtocolParameters in place
% of makeTreadmillSpeed_Split and set SplitOrTied to 'tied'. Units, ramps and
% outputs are as in makeTreadmillSpeed_Split, but no plot is made.
% Inputs:  paramFixed - uses speedProtocol

L = 0.95; g = 9.81;
timeScaling = sqrt(L/g);
timePerPhaseInMin = 2;

% Tied protocols use a 3-second transition regardless of the caller's value.
paramFixed.transitionTime = 3;

vNormal = -0.35;
vSlow = -0.35*0.75;
vVerySlow = -0.35*0.5;
vVeryFast = -0.35*1.5;
vFast = -0.35*1.25;

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

    case 'single speed change pulse'

        tDurationTransition = paramFixed.transitionTime/timeScaling;

        tDuration1 = 4*60/timeScaling;
        footSpeed1_phase1 = vNormal;
        footSpeed2_phase1 = vNormal;

        tDuration2 = 1*60/timeScaling;
        footSpeed1_phase2 = vNormal;
        footSpeed2_phase2 = vNormal;

        tDuration3 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase3 = vFast;
        footSpeed2_phase3 = vFast;

        tDuration4 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase4 = vNormal;
        footSpeed2_phase4 = vNormal;

        tStore = [0; tDuration1; tDuration2; tDuration3; tDuration4];
        tStore = cumsum(tStore);

        footSpeed1Store = [footSpeed1_phase1; footSpeed1_phase1; ...
            footSpeed1_phase2; footSpeed1_phase3; footSpeed1_phase4];
        footSpeed2Store = [footSpeed2_phase1; footSpeed2_phase1; ...
            footSpeed2_phase2; footSpeed2_phase3; footSpeed2_phase4];

    case 'single speed change'

        tDurationTransition = paramFixed.transitionTime/timeScaling;

        tDuration1 = 4*60/timeScaling;
        footSpeed1_phase1 = vNormal;
        footSpeed2_phase1 = vNormal;

        tDuration2 = 1*60/timeScaling;
        footSpeed1_phase2 = vNormal;
        footSpeed2_phase2 = vNormal;

        tDuration3 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase3 = vFast;
        footSpeed2_phase3 = vFast;

        tStore = [0; tDuration1; tDuration2; tDuration3];
        tStore = cumsum(tStore);

        footSpeed1Store = [footSpeed1_phase1; footSpeed1_phase1; ...
            footSpeed1_phase2; footSpeed1_phase3];
        footSpeed2Store = [footSpeed2_phase1; footSpeed2_phase1; ...
            footSpeed2_phase2; footSpeed2_phase3];

    case 'two speed changes'

        tDurationTransition = paramFixed.transitionTime/timeScaling;

        tDuration1 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase1 = vNormal;
        footSpeed2_phase1 = vNormal;

        tDuration2 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase2 = vFast;
        footSpeed2_phase2 = vFast;

        tDuration3 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase3 = vNormal;
        footSpeed2_phase3 = vNormal;

        tDuration4 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase4 = vSlow;
        footSpeed2_phase4 = vSlow;

        tDuration5 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase5 = vNormal;
        footSpeed2_phase5 = vNormal;

        tStore = [0; tDuration1; tDuration2; tDuration3; tDuration4; tDuration5];
        tStore = cumsum(tStore);

        footSpeed1Store = [footSpeed1_phase1; footSpeed1_phase1; ...
            footSpeed1_phase2; footSpeed1_phase3; footSpeed1_phase4; footSpeed1_phase5];
        footSpeed2Store = [footSpeed2_phase1; footSpeed2_phase1; ...
            footSpeed2_phase2; footSpeed2_phase3; footSpeed2_phase4; footSpeed2_phase5];

    case 'four speed changes'

        tDurationTransition = paramFixed.transitionTime/timeScaling;

        tDuration1 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase1 = vNormal;
        footSpeed2_phase1 = vNormal;

        tDuration2 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase2 = vSlow;
        footSpeed2_phase2 = vSlow;

        tDuration3 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase3 = vNormal;
        footSpeed2_phase3 = vNormal;

        tDuration4 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase4 = vFast;
        footSpeed2_phase4 = vFast;

        tDuration5 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase5 = vNormal;
        footSpeed2_phase5 = vNormal;

        tDuration6 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase6 = vVeryFast;
        footSpeed2_phase6 = vVeryFast;

        tDuration7 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase7 = vNormal;
        footSpeed2_phase7 = vNormal;

        tDuration8 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase8 = vVerySlow;
        footSpeed2_phase8 = vVerySlow;

        tDuration9 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase9 = vNormal;
        footSpeed2_phase9 = vNormal;

        tStore = [0; tDuration1; tDuration2; tDuration3; tDuration4; tDuration5; ...
            tDuration6; tDuration7; tDuration8; tDuration9];
        tStore = cumsum(tStore);

        footSpeed1Store = [footSpeed1_phase1; footSpeed1_phase1; ...
            footSpeed1_phase2; footSpeed1_phase3; footSpeed1_phase4; footSpeed1_phase5; ...
            footSpeed1_phase6; footSpeed1_phase7; footSpeed1_phase8; footSpeed1_phase9];
        footSpeed2Store = [footSpeed2_phase1; footSpeed2_phase1; ...
            footSpeed2_phase2; footSpeed2_phase3; footSpeed2_phase4; footSpeed2_phase5;
            footSpeed2_phase6; footSpeed2_phase7; footSpeed2_phase8; footSpeed2_phase9];

    case 'four speed changes 1'

        tDurationTransition = paramFixed.transitionTime/timeScaling;

        tDuration1 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase1 = vNormal;
        footSpeed2_phase1 = vNormal;

        tDuration2 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase2 = vFast;
        footSpeed2_phase2 = vFast;

        tDuration3 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase3 = vNormal;
        footSpeed2_phase3 = vNormal;

        tDuration4 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase4 = vSlow;
        footSpeed2_phase4 = vSlow;

        tDuration5 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase5 = vNormal;
        footSpeed2_phase5 = vNormal;

        tDuration6 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase6 = vVerySlow;
        footSpeed2_phase6 = vVerySlow;

        tDuration7 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase7 = vNormal;
        footSpeed2_phase7 = vNormal;

        tDuration8 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase8 = vVeryFast;
        footSpeed2_phase8 = vVeryFast;

        tDuration9 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase9 = vNormal;
        footSpeed2_phase9 = vNormal;

        tStore = [0; tDuration1; tDuration2; tDuration3; tDuration4; tDuration5; ...
            tDuration6; tDuration7; tDuration8; tDuration9];
        tStore = cumsum(tStore);

        footSpeed1Store = [footSpeed1_phase1; footSpeed1_phase1; ...
            footSpeed1_phase2; footSpeed1_phase3; footSpeed1_phase4; footSpeed1_phase5; ...
            footSpeed1_phase6; footSpeed1_phase7; footSpeed1_phase8; footSpeed1_phase9];
        footSpeed2Store = [footSpeed2_phase1; footSpeed2_phase1; ...
            footSpeed2_phase2; footSpeed2_phase3; footSpeed2_phase4; footSpeed2_phase5;
            footSpeed2_phase6; footSpeed2_phase7; footSpeed2_phase8; footSpeed2_phase9];

    case 'four speed changes 2'

        tDurationTransition = paramFixed.transitionTime/timeScaling;

        tDuration1 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase1 = vNormal;
        footSpeed2_phase1 = vNormal;

        tDuration2 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase2 = vVerySlow;
        footSpeed2_phase2 = vVerySlow;

        tDuration3 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase3 = vNormal;
        footSpeed2_phase3 = vNormal;

        tDuration4 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase4 = vFast;
        footSpeed2_phase4 = vFast;

        tDuration5 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase5 = vNormal;
        footSpeed2_phase5 = vNormal;

        tDuration6 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase6 = vSlow;
        footSpeed2_phase6 = vSlow;

        tDuration7 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase7 = vNormal;
        footSpeed2_phase7 = vNormal;

        tDuration8 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase8 = vVeryFast;
        footSpeed2_phase8 = vVeryFast;

        tDuration9 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase9 = vNormal;
        footSpeed2_phase9 = vNormal;

        tStore = [0; tDuration1; tDuration2; tDuration3; tDuration4; tDuration5; ...
            tDuration6; tDuration7; tDuration8; tDuration9];
        tStore = cumsum(tStore);

        footSpeed1Store = [footSpeed1_phase1; footSpeed1_phase1; ...
            footSpeed1_phase2; footSpeed1_phase3; footSpeed1_phase4; footSpeed1_phase5; ...
            footSpeed1_phase6; footSpeed1_phase7; footSpeed1_phase8; footSpeed1_phase9];
        footSpeed2Store = [footSpeed2_phase1; footSpeed2_phase1; ...
            footSpeed2_phase2; footSpeed2_phase3; footSpeed2_phase4; footSpeed2_phase5;
            footSpeed2_phase6; footSpeed2_phase7; footSpeed2_phase8; footSpeed2_phase9];

    case 'four speed changes 3'

        tDurationTransition = paramFixed.transitionTime/timeScaling;

        tDuration1 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase1 = vNormal;
        footSpeed2_phase1 = vNormal;

        tDuration2 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase2 = vVeryFast;
        footSpeed2_phase2 = vVeryFast;

        tDuration3 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase3 = vNormal;
        footSpeed2_phase3 = vNormal;

        tDuration4 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase4 = vFast;
        footSpeed2_phase4 = vFast;

        tDuration5 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase5 = vNormal;
        footSpeed2_phase5 = vNormal;

        tDuration6 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase6 = vVerySlow;
        footSpeed2_phase6 = vVerySlow;

        tDuration7 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase7 = vNormal;
        footSpeed2_phase7 = vNormal;

        tDuration8 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase8 = vSlow;
        footSpeed2_phase8 = vSlow;

        tDuration9 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase9 = vNormal;
        footSpeed2_phase9 = vNormal;

        tStore = [0; tDuration1; tDuration2; tDuration3; tDuration4; tDuration5; ...
            tDuration6; tDuration7; tDuration8; tDuration9];
        tStore = cumsum(tStore);

        footSpeed1Store = [footSpeed1_phase1; footSpeed1_phase1; ...
            footSpeed1_phase2; footSpeed1_phase3; footSpeed1_phase4; footSpeed1_phase5; ...
            footSpeed1_phase6; footSpeed1_phase7; footSpeed1_phase8; footSpeed1_phase9];
        footSpeed2Store = [footSpeed2_phase1; footSpeed2_phase1; ...
            footSpeed2_phase2; footSpeed2_phase3; footSpeed2_phase4; footSpeed2_phase5;
            footSpeed2_phase6; footSpeed2_phase7; footSpeed2_phase8; footSpeed2_phase9];

    case 'four speed changes 4'

        tDurationTransition = paramFixed.transitionTime/timeScaling;

        tDuration1 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase1 = vNormal;
        footSpeed2_phase1 = vNormal;

        tDuration2 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase2 = vFast;
        footSpeed2_phase2 = vFast;

        tDuration3 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase3 = vNormal;
        footSpeed2_phase3 = vNormal;

        tDuration4 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase4 = vVerySlow;
        footSpeed2_phase4 = vVerySlow;

        tDuration5 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase5 = vNormal;
        footSpeed2_phase5 = vNormal;

        tDuration6 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase6 = vVeryFast;
        footSpeed2_phase6 = vVeryFast;

        tDuration7 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase7 = vNormal;
        footSpeed2_phase7 = vNormal;

        tDuration8 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase8 = vSlow;
        footSpeed2_phase8 = vSlow;

        tDuration9 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase9 = vNormal;
        footSpeed2_phase9 = vNormal;

        tStore = [0; tDuration1; tDuration2; tDuration3; tDuration4; tDuration5; ...
            tDuration6; tDuration7; tDuration8; tDuration9];
        tStore = cumsum(tStore);

        footSpeed1Store = [footSpeed1_phase1; footSpeed1_phase1; ...
            footSpeed1_phase2; footSpeed1_phase3; footSpeed1_phase4; footSpeed1_phase5; ...
            footSpeed1_phase6; footSpeed1_phase7; footSpeed1_phase8; footSpeed1_phase9];
        footSpeed2Store = [footSpeed2_phase1; footSpeed2_phase1; ...
            footSpeed2_phase2; footSpeed2_phase3; footSpeed2_phase4; footSpeed2_phase5;
            footSpeed2_phase6; footSpeed2_phase7; footSpeed2_phase8; footSpeed2_phase9];

    case 'four speed changes 5'

        tDurationTransition = paramFixed.transitionTime/timeScaling;

        tDuration1 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase1 = vNormal;
        footSpeed2_phase1 = vNormal;

        tDuration2 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase2 = vVerySlow;
        footSpeed2_phase2 = vVerySlow;

        tDuration3 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase3 = vNormal;
        footSpeed2_phase3 = vNormal;

        tDuration4 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase4 = vVeryFast;
        footSpeed2_phase4 = vVeryFast;

        tDuration5 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase5 = vNormal;
        footSpeed2_phase5 = vNormal;

        tDuration6 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase6 = vFast;
        footSpeed2_phase6 = vFast;

        tDuration7 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase7 = vNormal;
        footSpeed2_phase7 = vNormal;

        tDuration8 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase8 = vSlow;
        footSpeed2_phase8 = vSlow;

        tDuration9 = timePerPhaseInMin*60/timeScaling;
        footSpeed1_phase9 = vNormal;
        footSpeed2_phase9 = vNormal;

        tStore = [0; tDuration1; tDuration2; tDuration3; tDuration4; tDuration5; ...
            tDuration6; tDuration7; tDuration8; tDuration9];
        tStore = cumsum(tStore);

        footSpeed1Store = [footSpeed1_phase1; footSpeed1_phase1; ...
            footSpeed1_phase2; footSpeed1_phase3; footSpeed1_phase4; footSpeed1_phase5; ...
            footSpeed1_phase6; footSpeed1_phase7; footSpeed1_phase8; footSpeed1_phase9];
        footSpeed2Store = [footSpeed2_phase1; footSpeed2_phase1; ...
            footSpeed2_phase2; footSpeed2_phase3; footSpeed2_phase4; footSpeed2_phase5;
            footSpeed2_phase6; footSpeed2_phase7; footSpeed2_phase8; footSpeed2_phase9];

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
