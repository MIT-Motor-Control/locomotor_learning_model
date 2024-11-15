
# Setting inital parameters 

class paramFixed:
    def __init__(self):        
        
        # setting Biped Model Parameters 

        self.includeAccelerationTorque = 1
        self.mbody = 1 
        self.leglength = 1 
        self.gravg = 1

        # efficiency of positive and negative work
        self.efficiency_neg = 1.2 
        self.efficiency_pos = 0.25
        self.bPos = 1/self.efficiency_pos
        self.bNeg = 1/self.efficiency_neg

        # swing leg energy cost parameters
        self.mFoot = 0.05
        self.swingCost_Coeff = 0.9
        self.swingCost_alpha = 1.0

        # energy vs periodicity variance reduction
        self.lambdaEnergyVsPeriodicity = 1      # 1 = energy, 0 = periodicity

        # energy vs symmetry
        self.lambdaEnergyVsSymmetry = 0.75      # 1 = energy, 0 = symmetry 
        self.symmetryMultiplier = 10        # just multiplies the symmetry objective
        
    def print_all_variables(self):
        for attr, value in self.__dict__.items():
            print(f"{attr}: {value}")
