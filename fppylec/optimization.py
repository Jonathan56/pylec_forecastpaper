from pyomo.opt import SolverFactory
from pyomo.environ import *
import pandas


def pool(f_houses_kW, f_pv_kW, extra):
    """Common pool optimal battery control

    Note : Energy at midnight is calculated with power at 23:45, power at midnight is forced to zero with endby = len(df) - 1, so there is no impact to add one last value to df (representing midnight). It allows to really define end_kWh for 00:00 and avoid losing 15 minutes.
    """

    # Format uncontrollable demand
    df = pandas.DataFrame(index=list(range(0, len(f_houses_kW))),
                          data={'p': (f_houses_kW - f_pv_kW).tolist()})
    df.loc[len(df), 'p'] = 0  # Append one value to go to midnight
    assert len(df) == len(f_houses_kW) + 1

    # Option to ommit start by and end by
    if 'startby' not in extra:
        extra['startby'] = 0
    if 'endby' not in extra:
        extra['endby'] = len(df)-1

    # Order book for batteries
    data = {'startby': [extra['startby']],
            'endby': [extra['endby']],  # P_tlast=0
            'min_kw': [extra['battery_kW']],
            'max_kw': [extra['battery_kW']],
            'max_kwh': [extra['battery_kWh']],
            'initial_kwh': extra['initial_kwh'],
            'end_kwh': [extra['battery_kWh'] / 2],
            'eta': [extra['eta']]}
    dfbatteries = pandas.DataFrame(data=data)

    # Run optimization (00:00 to 00:00)
    glpk = maximize_self_consumption_vforecast(
        df, dfbatteries, timestep=15/60, solver='glpk')

    # Get SOC for each battery at 00:00 (index=96, len=97) on the first day
    SOC_end = glpk['batteryenergy'].loc[96,:].tolist()

    # Return results, remove power at midnight (0 in anycase)
    assert glpk['demand_controllable'][-1] == 0
    return (glpk['demand_controllable'][:-1],
            {'SOC_end' : SOC_end})

def maximize_self_consumption_vforecast(
        uncontrollable, dfbatteries, dfshapeables=None, dfdeferrables=None,
        timestep=15/60, solver='glpk', verbose=False, solver_path=None,
        timelimit=5*60):
    """
    Version vforecast Minimize \sum_{t}^T p_{import}(t)
    Optimize batteries, to maximize collective self-consumption.
    Inputs:
        - uncontrollable (DataFrame): uncontrollable load demand "p"
        - dfbatteries (DataFrame): order book
        - timestep (float): one is equivalent to hourly timestep

        # Net demand
        uncontrollable['p'] = uncontrollable['houses_kW'] - uncontrollable['pv_kW']
        uncontrollable['index'] = list(range(0, len(uncontrollable)))
        uncontrollable.set_index('index', inplace=True)
        uncontrollable = uncontrollable[['p']]

        # Order book for battery
        data = {'startby': [0],
                'endby': [len(uncontrollable)],
                'min_kw': [battery_kW],
                'max_kw': [battery_kW],
                'max_kwh': [battery_kWh],
                'initial_kwh': [battery_kWh / 2] ,
                'end_kwh': [battery_kWh / 2],
                'eta': [0.95]}
        dfbatteries = pandas.DataFrame(data=data)

    Outputs:
        - batteryenergy
        - demand_controllable
    """
    # Inputs
    horizon = uncontrollable.index.tolist()
    demand_uncontrollable = uncontrollable.p.to_list()
    batteries = dfbatteries.index.tolist()
    m = ConcreteModel()

    ###################################################### Set
    m.horizon = Set(initialize=horizon, ordered=True)
    last = m.horizon.last()
    m.batteries = Set(initialize=batteries, ordered=True)

    ##################################################### Var
    m.community_import = Var(m.horizon, domain=Reals)
    m.demand_controllable = Var(m.horizon, domain=Reals)

    # Equipment specifics
    m.batteryin = Var(m.horizon, m.batteries, domain=Reals)
    m.batteryout = Var(m.horizon, m.batteries, domain=Reals)
    m.batteryenergy = Var(m.horizon, m.batteries, domain=Reals)

    #################################################### Rules
    # --------------------------------------------------------
    # ---------------------Battery----------------------------
    # --------------------------------------------------------
    # The power bounds are defined by the battery characteristics
    def r_battery_min_powerin(m, t, b):
        return (m.batteryin[t, b] >= 0)

    def r_battery_max_powerin(m, t, b):
        return (m.batteryin[t, b] <= dfbatteries.loc[b, 'max_kw'])

    def r_battery_min_powerout(m, t, b):
        return (m.batteryout[t, b] >= 0)

    def r_battery_max_powerout(m, t, b):
        return (m.batteryout[t, b] <= dfbatteries.loc[b, 'min_kw'])

    # Define the SOC considering charge/discharge efficiency
    # Use previous power demand to define current energy
    def r_battery_energy(m, t, b):
        if t == 0:
            return m.batteryenergy[t, b] == dfbatteries.loc[b, 'initial_kwh']
        else:
            return (m.batteryenergy[t, b] ==
                    m.batteryenergy[t-1, b] +
                    m.batteryin[t-1, b]  # !! at first we used P_t (cf paper)
                    * timestep * dfbatteries.loc[b, 'eta']
                    - m.batteryout[t-1, b]
                    * timestep / dfbatteries.loc[b, 'eta'])
                    # 0.25 pour un quart d'heure

    # Energy bound during operation
    def r_battery_min_energy(m, t, b):
        return (m.batteryenergy[t, b] >= 0)

    def r_battery_max_energy(m, t, b):
        return (m.batteryenergy[t, b] <= dfbatteries.loc[b, 'max_kwh'])

    # Energy status at the end
    def r_battery_end_energy(m, b):
        return (m.batteryenergy[last, b] >= dfbatteries.loc[b, 'end_kwh'])

    # If we are outside of startby - endby, we enforce no operation
    def r_batteryin_timebounds(m, t, b):
        if t < dfbatteries.loc[b, 'startby']:
            return (m.batteryin[t, b] == 0)
        if t >= dfbatteries.loc[b, 'endby']:  # >= help with P_tlast = 0
            return (m.batteryin[t, b] == 0)
        else:
            return Constraint.Skip

    def r_batteryout_timebounds(m, t, b):
        if t < dfbatteries.loc[b, 'startby']:
            return (m.batteryout[t, b] == 0)
        if t >= dfbatteries.loc[b, 'endby']:  # >= help with P_tlast = 0
            return (m.batteryout[t, b] == 0)
        else:
            return Constraint.Skip

    # --------------------------------------------------------
    # ---------------------Helper-----------------------------
    # --------------------------------------------------------
    # Useless step which seems to be necessary
    def r_demand_total(m, t):
        return (m.demand_controllable[t] ==
                sum(m.batteryin[t, b] - m.batteryout[t, b] for b in m.batteries))

    # Community import (by definition above zero)
    def community_import_zero(m, t):
        return (m.community_import[t] >= 0)

    # Community import larger than contr + unctr
    def community_import(m, t):
        return (m.community_import[t] >=
                m.demand_controllable[t] + demand_uncontrollable[t])

    # --------------------------------------------------------
    # ---------------------Add To Model-----------------------
    # --------------------------------------------------------
    # Battery
    m.r5 = Constraint(m.horizon, m.batteries, rule=r_battery_min_powerin)
    m.r6 = Constraint(m.horizon, m.batteries, rule=r_battery_max_powerin)
    m.r7 = Constraint(m.horizon, m.batteries, rule=r_battery_min_powerout)
    m.r8 = Constraint(m.horizon, m.batteries, rule=r_battery_max_powerout)
    m.r9 = Constraint(m.horizon, m.batteries, rule=r_battery_energy)
    m.r10 = Constraint(m.horizon, m.batteries, rule=r_battery_min_energy)
    m.r11 = Constraint(m.horizon, m.batteries, rule=r_battery_max_energy)
    m.r12 = Constraint(m.batteries, rule=r_battery_end_energy)
    m.r13 = Constraint(m.horizon, m.batteries, rule=r_batteryin_timebounds)
    m.r14 = Constraint(m.horizon, m.batteries, rule=r_batteryout_timebounds)
    # Helper
    m.r15 = Constraint(m.horizon, rule=r_demand_total)
    m.r23 = Constraint(m.horizon, rule=community_import_zero)
    m.r24 = Constraint(m.horizon, rule=community_import)

    ##################################################### Objective function
    # Linear objective function
    def objective_function(m):
        return sum(m.community_import[i] for i in m.horizon)
    m.objective = Objective(rule=objective_function, sense=minimize)

    #################################################### Run
    # Solve optimization problem
    with SolverFactory(solver, executable=solver_path) as opt:
        if solver in 'glpk':
            opt.options['tmlim'] = timelimit
            results = opt.solve(m, tee=verbose)
        if solver in 'gurobi':
            opt.options['TimeLimit'] = timelimit
            results = opt.solve(m, tee=verbose)
        if solver in 'cbc':
            results = opt.solve(m, timelimit=timelimit, tee=verbose)
        else:
            raise NotImplementedError
    if verbose:
        print(results)

    #################################################### Results
    # A Dictionnary contains all the results
    results = {}
    keys = ['batteryenergy']

    for key in keys:
        try:
            tmp = pandas.DataFrame(index=['none'],
                    data=getattr(m, key).get_values())
            tmp = tmp.transpose()
            tmp = tmp.unstack(level=1)
            tmp.columns = tmp.columns.levels[1]
            results[key] = tmp.copy()
        except:
            results[key] = None

    # demand_controllable
    results['demand_controllable'] = list(
        m.demand_controllable.get_values().values())
    return results
