import pandas as pd
from datetime import datetime

"""
John's runModel function

Modified by Braeden Fieguth, "Export Model Results" section by Dave Innes

runModel(
    Start, runTime, integInt, communInt, 
    prev_output,
    output_file, filename, filepath, fileext
    ), where
    Start = 0 (to start from time = 0), or Start = 1 (to continue from previous run)
    runTime = duration of simulation in user-defined time units
    integInt = integration interval in user-defined time units
    communInt = time interval for communicating results to csv file

    outputs_list = list, names of variables to include in output
    parameters = dict, dictionary with all model parameters
    initial_stateVars = list, all initial state variables
    model_function = function, name of function with model equations

    prev_output = if Start == 1 then must specificy the dataframe that was output from previous run

    output_file = True/False - should output be exported to a csv file? Default is False.
    filepath = file path to the folder to save the file in. Default is './' (which means current directory)
    filename = name of the file (without extension). Date/time automatically added after this name. Default is 'generic'
    fileextension = extension to save file with. Default is '.csv'

"""
def runModel(Start, runTime, integInt, communInt,
             outputs_list,
             parameters,
             initital_stateVars,
             model_function,
             prev_output= None,
             output_file = False,
             filepath = './',
             filename = 'generic',
             fileextension = '.csv'
             ):
    
    ### Setup Integration and Communication Loop ###
    lastIntervalNo=runTime/integInt 
    intervalNoForComm=communInt/integInt
    cintAsInt=int(intervalNoForComm)

    ### Initialize Lists ###
    model_results = []  # Store model results, a list of lists
    slopes=[] # list to hold slopes (diff eqn results) for Runge-Kutta
    start=[] # list to hold stateVar values at beginning of current integInt
    
    print("Running Model....")

    ####################
    # Start New Simulation
    ####################
    if Start==0: # start from t=0 instead of continue from where it left off
        t=0.0 # start time for simulation
        # Run model at time=0, uses initial state variables that user input
        differential_return, variable_returns = model_function(parameters=parameters,
                                                                stateVars=initital_stateVars,
                                                                outputs_list=outputs_list,
                                                                t=t
                                                                ) 
        model_results.append(variable_returns)        
            # dynamic() now returns a list of variables that can be appended into model_results

    ####################
    # Continue Simulation
    ####################
    if Start == 1:
        # check that prev_output has been included
        if not isinstance(prev_output, pd.DataFrame):
            raise TypeError("The variable prev_output must be a dataframe if Start == 1")

    # 4th-order Runge Kutta algorithm to iterate through dynamic() between t = 0 and tStop
    stateVars = initital_stateVars.copy()
        # Create copy of the initial state variables 
    for intervalNo in range(int(lastIntervalNo)):
        for n in range(4): # 4 parts to Runge-Kutta estimation of new state
            # eval model fluxes and store diff eqn results in slopes[part][statevar] for 
            # each part of Runge-Kutta by calling dynamic() here:
            differential_return, variable_returns = model_function(parameters=parameters,
                                                                    stateVars=stateVars,
                                                                    outputs_list=outputs_list,
                                                                    t=t)
            slopes.append(differential_return)  # Add list of differentials
            for svno in range(len(stateVars)): # estimate new stateVar values 1 at a time
                match n:
                    case 0: # if part 1 of Runge-Kutta, record state at beginning of integInt
                        start.append(stateVars[svno]) # to be used throughout
                        newStateVar=start[svno]+integInt*slopes[n][svno]/2
                    case 1: # if part 2 of Runge Kutta
                        newStateVar=start[svno]+integInt*slopes[n][svno]/2
                    case 2: # if part 3 of Runge Kutta
                        newStateVar=start[svno]+integInt*slopes[n][svno]
                    case 3: # if part 4 of Runge-Kutta, use all 4 slopes to estimate new state
                        newStateVar=start[svno]+integInt/6*(slopes[0][svno]+2*slopes[1][svno]+
                            2*slopes[2][svno]+slopes[3][svno])
                        
                stateVars[svno]=newStateVar 
                    # update stateVars w new values after each part of Runge-Kutta:
        # end of one iteration thru complete 4th-order Runge-Kutta algorithm = 1 integInt
        t+=integInt 
            # increment time to associate with new state
        # output results of new state if new time is a communication time
        # current intervalNo is associated with old time so intervalNo+1 is used
        remainder=(intervalNo+1)/cintAsInt-int((intervalNo+1)/cintAsInt)
        if remainder==0:
            model_results.append(variable_returns.copy()) # appends pointer to temp outputRow so need to append copy
        slopes.clear() # clear slopes list for next integInt
        start.clear() # clear temp list for next integInt

    ####################
    # Export Model Results
    ####################
    output_dataframe = pd.DataFrame(model_results, columns = outputs_list)

    if Start == 1:
        # join new data onto previous data
        output_dataframe = pd.concat([prev_output,output_dataframe], ignore_index=True)

    if output_file == True:
        # get current date/time as string
        timestamp = datetime.datetime.now().strftime('%y%m%d_%H%M%S') 
        
        # put together full filename and path
        full_filename = filepath + filename + timestamp + fileextension
        print("File saved to: " + full_filename)

        # save to csv
        output_dataframe.to_csv(full_filename, index=False)
    
    print(output_dataframe)

    print("Running Model....DONE")

    # return the dataframe to be used later
    return output_dataframe
