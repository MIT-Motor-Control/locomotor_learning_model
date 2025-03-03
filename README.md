# locomotor_learning_model

------- Supplementary code to the manuscript ---------------------------------

Title: "Exploration-based learning of a stabilizing controller predicts locomotor adaptation"
Authors: Nidhi Seethapathi, Barrett L Clark, and Manoj Srinivasan

This code version is offered for review purposes. Upon acceptance/publication 
of the manuscript, the entire code will be made available without any 
restrictions in a public repository (e.g., github).

------- How to run the program -----------------------------------------------

The following assumes that you have downloaded the zip file onto your 
computer, and have un-zipped the .zip file to create a folder. The unzipped 
folder will be programsForManuscriptSeethapathiEtAl22 by default.

If you have MATLAB already installed on your computer, please directly run the 
following program within MATLAB: 
	rootSimulateLearningWhileWalking.m
This program calls other functions as necessary to perform the simulation, and 
will generate the key figures. 

To run the program in MATLAB, navigate to the 
folder containing the code, and type rootSimulateLearningWhileWalking into the
command prompt and press 'enter' or 'return':
>> rootSimulateLearningWhileWalking <enter/return>

If don't have MATLAB on your computer or if the above step does not work, 
see the next section and come back to this section.

------- Programming enviroment and computer requirements ---------------------

The program is written from scratch in the MATLAB programming environment.

The program is open source in the sense you can view the entire code here.

The code is self-contained and does not use any special functions. The only 
toolbox that may possibly need to be installed is the MATLAB Optimization 
toolbox, but this may not be necessary depending on your MATLAB installation.

The programs were tested in MATLAB Version: 9.12.0.1927505 (R2022a). However, 
the programs will run in any older version of MATLAB, as the programs do not 
require any special functionality available only in the newer versions. 

MATLAB is available from Mathworks: 
https://www.mathworks.com

One may use the program either on a personal computer (eg., Desktop or 
Laptop) or entirely on the cloud through a browser via MATLAB online:
https://www.mathworks.com/products/matlab-online.html#license-types

Octave is a free clone of MATLAB and is available here under the 
GNU General Public License (GPL).
https://octave.org
MATLAB programs that don't use special toolboxes, as here, can be run
on Octave essentially verbatim. You should be able to run our programs
on Octave.

------- Other general notes --------------------------------------------------


1) The simulation parameters and conditions are encoded within the various 
loadParameters*** files. 

2) See the Methods section and Supplementary Appendix of the manuscript for 
the mathematical equations underlying the simulation.

3) Because the simulation is stochastic, running the program two times will
give slightly different curves, while having broad statistical similarity.

4) If your program hangs for whatever reason, please exit the program via 
crtl-c or exit MATLAB through a force quit, and then re-start the program.
The qualitative results are invariant to substantial changes to the learning 
parameters. But if the parameters are outside of the stability region (as
discussed in the manuscript), the biped or the learner may go unstable, 
resulting in numerical challenges that may result in the program hanging, as in
case-H below.

5) The following are some things to try to examine different aspects of the
results, as a quick demonstration. This list is by no means comprehensive, as 
the code allows infinite numerical experiments; these are suggested for 
qualitative understanding of the results and the structure of the code.

- A) Run the program provided as is. This is the default version of the code. 
It will run a classic split-belt adaptation protocol: tied belt, split belt, and 
tied belt again.

- B) Turn off learning and memory by setting the following parameters to 
zero. To do this, go to loadLearnerParameters.m and set the following 
parameters to zero: paramFixed.Learner.LearningRate and 
paramFixed.Learner.LearningRateTowardMemory. You will find that there is no
gradual slow-timescale adaptation: the step length asymmetry looks piecewise
constant. No learning, just feedback control response.

- C) Go back to the default version of the code. Now just set 
paramFixed.Learner.LearningRateTowardMemory = 0 in loadLearnerParameters.m 
No memory use, so deadaptation is not particularly faster than adaptation.

- E) Go back to the default version of the code, which has non-zero 
paramFixed.Learner.LearningRateTowardMemory, e.g., 0.02, You will see that 
deadaptation transient is faster than adaptation transient.
In the future, we will release a version of the code that has options to 
directly demonstrate these and many other parameter settings as opposed to
having to manually change the parameters.

- F) Go back to the default version of the code. To see degraded learning or 
biped/learning instability, increase sensory noise in the 
loadSensoryNoiseParameters.m program, or increase/decrease the learning rates in loadLearnerParameters.m. See remark 6 above about instabilities.