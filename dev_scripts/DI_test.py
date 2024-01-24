import modelling_tools as mt

### UNITS ###
# all units are: mmol, L, h
# G = glucose; A = Acetate; F = Fatty Acids; X = Amino Acids
# L = Lactose; C = CO2; F = Fat; P = Protein


params = {
    'vol': 2.0,
    'kOG': 126.3, 'kGC': 544, 'kGL': 718,
    'kOA': 231.3, 'kAF': 1794.3, 'kFA': 1615, 'kAC': 2737,
    'kOF': 143.3, 'kFF': 1566,
    'kOX': 96.19, 'kXP': 434.1, 'kXC': 58.5
}

# set initial values of state variables
iStateVars = {
    #mM
    'G' : 0.54, # Glucose
    'A' : 0.7,  # Acetate
    'F' : 0.106, # Fatty acids
    'X' : 0.82 # Amino Acids
    } 



outputs_list = ['t', 'G', 'A', 'F', 'X', 
               'concG', 'concA', 'concF', 'concX', 
               'PGOG', 'UGGC', 'UAAF', 'UGGL', 'PAOA', 'UFFA', 'PAFA', 'UAAC', 'PFOF', 'PFAF', 'UFFF', 'PXOX', 'UXXP', 'UXXC', 
               'dGdt', 'dAdt', 'dFdt', 'dXdt']



differential_variables = ['dGdt', 'dAdt', 'dFdt', 'dXdt']

def milkcomp_assign_model(
                parameters, 
                stateVars,  
                t # does t need to be here?
                ):
            
        # Assign Parameter Values #

        vol = parameters['vol']

        #set arterial concentrations as constants (mM)
        arterialG = 2.7; arterialA = 3.9; arterialF = 0.3; arterialX = 2.1

        # Variables w/ Differential Equation #
        # A = stateVars[0]
        # B = stateVars[1]

        # this doesnt' work as would be expected, because although we give a dictionary
        # to intiailise model, it then gets converted to a list when it is called when run 
        # e.g. self.input_model(parameters=self.parameters,
                            #  stateVars=self.istate_vars,          <- this is a list
                            #  t=t) 
        # concG = stateVars['G'] / vol
        # concA = stateVars['A'] / vol
        # concF = stateVars['F'] / vol
        # concX = stateVars['X'] / vol
       
       # This also doesn't work, because it is somehow expecting the same names from dict to be 
        # in this functions environment:
        # concG = stateVars[0] / vol
        # concA = stateVars[1] / vol
        # concF = stateVars[2] / vol
        # concX = stateVars[3] / vol

        # the following works, from original python format, but means there's no point using a dictionary above.

        # list all vars that have a diff eqn
        G = stateVars[0] # Glucose
        A = stateVars[1] # Acetate
        F = stateVars[2]  # Fatty Acids
        X = stateVars[3] # Amino Acids

        # the following model eqns must be in calculation order
        concG = G / vol
        concA = A / vol
        concF = F / vol
        concX = X / vol
        
        # Glucose
        PGOG = arterialG * parameters['kOG']
        UGGC = concG * parameters['kGC']
        UGGL = concG * parameters['kGL']

        # Acetate
        PAOA = arterialA * parameters['kOA']
        UAAF = concA * parameters['kAF']
        UFFA = concF * parameters['kFA']
        PAFA = UFFA * 8 # 1 FA produces 8 Acetates
        UAAC = concA * parameters['kAC']

        # Fatty Acids
        PFOF = arterialF * parameters['kOF']
        PFAF = UAAF / 5 # 5 Ac used to make 1 Fa
        #UFFA above
        UFFF = concF * parameters['kFF']

        # AA (X)
        PXOX = arterialX * parameters['kOX']
        UXXP = concX * parameters['kXP']
        UXXC = concX * parameters['kXC']

        # diff eqns, 1 for each statevar
        dGdt = PGOG - UGGC - UGGL
        dAdt = PAOA + PAFA - UAAF - UAAC
        dFdt = PFOF + PFAF - UFFA - UFFF
        dXdt = PXOX - UXXP - UXXC
    
        # Would be nice if this doesn't need to be defined here, otherwise can't use it in different instances of Model()
        # kind of circular argument giving a function that calls a class to the class which calls the function
        di_test.save_results()


di_test = mt.Model(
        parameters=params,
        istate_vars=iStateVars,
        outputs_list=outputs_list,
        input_model=milkcomp_assign_model,
        differential_variables=differential_variables
                   )


# run model
di_test.runModel(0, 0.1, 0.001, 0.01 )

df_output = di_test.prev_output

df_output



