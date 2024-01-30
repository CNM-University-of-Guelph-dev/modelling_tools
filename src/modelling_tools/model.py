import pandas as pd
import inspect

class Model:
    def __init__(self, parameters, istate_vars, outputs_list, input_model, differential_variables):
        self.parameters = parameters
        self.outputs_list = outputs_list
        self.input_model = input_model
        self.differential_variables = differential_variables
        self.istate_vars = istate_vars       
        # Store data from runModel
        self.differential_return = None
        self.variable_returns = None
        self.prev_output = None
                

    def runModel(self, 
                 runTime, 
                 integInt, 
                 communInt,
                 continue_run = False
                ):
        """
        Run a simulation using the 4th-order Runge-Kutta method and return the results as a dataframe and/or a CSV file.

        Parameters
        ----------
        Start : int
            Flag to determine whether to start a new simulation (0) or continue from a previous run (1).
        runTime : float
            Duration of the simulation in user-defined time units.
        integInt : float
            Integration interval in user-defined time units.
        communInt : float
            Time interval for communicating results to a dataframe.
        outputs_list : list
            Names of variables to include in the output.
        parameters : dict
            Dictionary with all model parameters.
        initial_stateVars : list
            List of all initial state variables.
        model_function : function
            Function containing model equations.
        prev_output : pd.DataFrame, optional
            Dataframe containing results from a previous run. Required if Start is 1.
        output_file : bool, optional
            Should output be exported to a CSV file? Default is False.
        filepath : str, optional
            File path to the folder to save the file in. Default is './' (current directory).
        filename : str, optional
            Name of the file (without extension). Date/time is automatically added after this name. Default is 'generic'.
        fileextension : str, optional
            Extension to save the file with. Default is '.csv'.

        Returns
        -------
        pd.DataFrame
            Dataframe containing the model results.

        Notes
        -----
        - The function uses the 4th-order Runge-Kutta algorithm for simulation.
        - If Start is 0, the simulation starts from time t=0; if Start is 1, it continues from the previous run.
        - If output_file is True, the results are exported to a CSV file.

        Examples
        --------
        To start a new simulation:

        >>> output_data = runModel(Start=0, runTime=10, integInt=0.1, communInt=1,
        ...                        outputs_list=['variable1', 'variable2'],
        ...                        parameters={'param1': 0.5, 'param2': 1.0},
        ...                        initial_stateVars=[1.0, 2.0],
        ...                        model_function=my_model_function)

        To continue a previous simulation:

        >>> output_data = runModel(Start=1, runTime=5, integInt=0.1, communInt=1,
        ...                        outputs_list=['variable1', 'variable2'],
        ...                        parameters={'param1': 0.5, 'param2': 1.0},
        ...                        initial_stateVars=[1.0, 2.0],
        ...                        model_function=my_model_function,
        ...                        prev_output=previous_results_df)
        """
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
        if not continue_run: # start from t=0 instead of continue from where it left off
            t=0.0 # start time for simulation
            istate_vars_import = list(self.istate_vars.values())
            stateVars_list = istate_vars_import.copy()
            stateVars_dict = self.istate_vars

            # Create copy of the initial state variables 
            # Run model at time=0, uses initial state variables that user input
            values = self.input_model(parameters=self.parameters,
                                      stateVars=self.istate_vars,
                                      t=t)
            differential_return = [values[var] for var in self.differential_variables]
            variable_returns = [values[variable_name] for variable_name in self.outputs_list]
            self.differential_return = differential_return
            self.variable_returns = variable_returns
                             
            model_results.append(self.variable_returns)        
                # dynamic() now returns a list of variables that can be appended into model_results

        ####################
        # Continue Simulation
        ####################
        else:   # If continue_run = True continue from previous timepoint
            # check that prev_output has been included
            if not isinstance(self.prev_output, pd.DataFrame):              
                raise TypeError("The variable prev_output must be a dataframe if Start == 1")
            t = self.prev_output['t'].iloc[-1]          

            stateVars_list = self.prev_output.iloc[-1, self.prev_output.columns.isin(self.istate_vars.keys())].tolist()
            stateVars_dict = self.prev_output.iloc[-1, self.prev_output.columns.isin(self.istate_vars.keys())].to_dict()
            # Extract the last row of 'prev_output' corresponding to the keys in self.istate_vars
   
        ####################
        # 4th-order Runge Kutta
        ####################
        # 4th-order Runge Kutta algorithm to iterate through dynamic() between t = 0 and tStop
        for intervalNo in range(int(lastIntervalNo)):
            for n in range(4): # 4 parts to Runge-Kutta estimation of new state
                # eval model fluxes and store diff eqn results in slopes[part][statevar] for 
                # each part of Runge-Kutta by calling dynamic() here:
                values = self.input_model(parameters=self.parameters,
                                          stateVars=stateVars_dict,
                                          t=t)
                differential_return = [values[var] for var in self.differential_variables]
                variable_returns = [values[variable_name] for variable_name in self.outputs_list]
                self.differential_return = differential_return
                self.variable_returns = variable_returns

                slopes.append(self.differential_return) # Add list of differentials
                for svno in range(len(stateVars_list)): # estimate new stateVar values 1 at a time
                    match n:
                        case 0: # if part 1 of Runge-Kutta, record state at beginning of integInt
                            start.append(stateVars_list[svno]) # to be used throughout
                            newStateVar = start[svno] + integInt * slopes[n][svno] / 2
                        case 1: # if part 2 of Runge Kutta
                            newStateVar=start[svno]+integInt*slopes[n][svno]/2
                        case 2: # if part 3 of Runge Kutta
                            newStateVar=start[svno]+integInt*slopes[n][svno]
                        case 3: # if part 4 of Runge-Kutta, use all 4 slopes to estimate new state
                            newStateVar=start[svno]+integInt/6*(slopes[0][svno]+2*slopes[1][svno]+
                                2*slopes[2][svno]+slopes[3][svno])
                            
                    stateVars_list[svno]=newStateVar 
                        # update stateVars w new values after each part of Runge-Kutta:
                    
                stateVars_dict.update(dict(zip(self.istate_vars.keys(), stateVars_list)))

            # end of one iteration thru complete 4th-order Runge-Kutta algorithm = 1 integInt
            t+=integInt 
                # increment time to associate with new state
            # output results of new state if new time is a communication time
            # current intervalNo is associated with old time so intervalNo+1 is used
            remainder=(intervalNo+1)/cintAsInt-int((intervalNo+1)/cintAsInt)
            if remainder==0:
                model_results.append(self.variable_returns.copy()) # appends pointer to temp outputRow so need to append copy
            
            slopes.clear() # clear slopes list for next integInt
            start.clear() # clear temp list for next integInt

        ####################
        # Export Model Results
        ####################
        output_dataframe = pd.DataFrame(model_results, columns = self.outputs_list)

        if continue_run:
            # join new data onto previous data
            output_dataframe = pd.concat([self.prev_output,output_dataframe], ignore_index=True)
       
        print(output_dataframe)
        self.prev_output = output_dataframe # Save output dataframe to be used again
        print("Running Model....DONE")
  