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


# --------------------------------------------------------
# ---------------Original optimization--------------------
# --------------------------------------------------------
def maximize_self_consumption_original(
        uncontrollable, dfbatteries, dfshapeables=None, dfdeferrables=None,
        timestep=15/60, solver='glpk', verbose=False, solver_path=None,
        timelimit=5*60):
    """
    Version v001 Minimize \sum_{t}^T peak^+ - peak^-
    Optimize batteries, shapeable and deferrable loads to maximize
    collective self-consumption.
    Inputs:
        - uncontrollable (DataFrame): uncontrollable load demand "p"
        - dfbatteries (DataFrame): order book
        - dfshapeables (DataFrame): order book
        - dfdeferrables (DataFrame): order book
        - timestep (float): one is equivalent to hourly timestep

        # Net demand
        uncontrollable['p'] = uncontrollable['houses_kW'] - uncontrollable['pv_kW']
        uncontrollable['index'] = list(range(0, len(uncontrollable)))
        uncontrollable.set_index('index', inplace=True)
        uncontrollable = uncontrollable[['p']]

        # Order book for Shapeable
        data = {'startby': [],
                'endby': [],
                'max_kw': [],
                'end_kwh': []}
        dfshapeables = pandas.DataFrame(data=data)

        # Order book for Deferrable
        data = {'startby': [],
                'endby': [],
                'duration': [],
                'profile_kw': []}
        dfdeferrables = pandas.DataFrame(data=data)

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
        - demandshape
        - batteryin
        - batteryout
        - batteryenergy
        - demanddeferr
        - deferrschedule
        - demand_controllable
        - community_import
        - total community_import
        - peakhigh
        - peaklow
    """
    # Inputs
    horizon = uncontrollable.index.tolist()
    demand_uncontrollable = uncontrollable.p.to_list()
    batteries = dfbatteries.index.tolist()
#     shapeables = dfshapeables.index.tolist()
#     deferrables = dfdeferrables.index.tolist()
    m = ConcreteModel()

    ###################################################### Set
    m.horizon = Set(initialize=horizon, ordered=True)
    last = m.horizon.last()
    m.batteries = Set(initialize=batteries, ordered=True)
#     m.shapeables = Set(initialize=shapeables, ordered=True)
#     m.deferrables = Set(initialize=deferrables, ordered=True)

    ##################################################### Var
    m.community_import = Var(m.horizon, domain=Reals)
    m.demand_controllable = Var(m.horizon, domain=Reals)
#     m.peakhigh = Var(domain=Reals)
#     m.peaklow = Var(domain=Reals)

    # Equipment specifics
#     m.demandshape = Var(m.horizon, m.shapeables, domain=Reals)
    m.batteryin = Var(m.horizon, m.batteries, domain=Reals)
    m.batteryout = Var(m.horizon, m.batteries, domain=Reals)
    m.batteryenergy = Var(m.horizon, m.batteries, domain=Reals)
#     m.demanddeferr = Var(m.horizon, m.deferrables, domain=Reals)
#     m.deferrschedule = Var(m.horizon, m.deferrables,
#                            domain=NonNegativeIntegers)

#     #################################################### Rules
#     # --------------------------------------------------------
#     # ------------------shapeable load------------------------
#     # --------------------------------------------------------
#     # The power bounds are defined by the load characteristics
#     def r_shape_min_power(m, t, s):
#         return (m.demandshape[t, s] >= 0)

#     def r_shape_max_power(m, t, s):
#         return (m.demandshape[t, s] <=
#                 dfshapeables.loc[s, 'max_kw'])

#     # At the end the energy asked by the load is satisfied
#     def r_shape_energy(m, s):
#         return (sum(m.demandshape[i, s] for i in m.horizon) ==
#                 dfshapeables.loc[s, 'end_kwh'])

#     # If we are outside of startby - endby, we enforce zero power
#     def r_shape_timebounds(m, t, s):
#         if t < dfshapeables.loc[s, 'startby']:
#             return m.demandshape[t, s] == 0
#         if t > dfshapeables.loc[s, 'endby']:
#             return m.demandshape[t, s] == 0
#         else:
#             return Constraint.Skip

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

    # # Define the SOC considering charge/discharge efficiency
    # Super slow (x5)
    # def r_battery_energy(m, t, b):
    #     return (m.batteryenergy[t, b] ==
    #             dfbatteries.loc[b, 'initial_kwh'] +
    #             sum(m.batteryin[k, b] for k in range(0, t))
    #             * timestep * dfbatteries.loc[b, 'eta']
    #             - sum(m.batteryout[k, b] for k in range(0, t))
    #             * timestep / dfbatteries.loc[b, 'eta'])

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

#     # --------------------------------------------------------
#     # ---------------------Deferrable-------------------------
#     # --------------------------------------------------------
#     # Convolution of the power profile (time horizon L)
#     # and the scheduler (time horizon T)
#     def r_deferrable_schedule(m, t, d):
#         return (m.demanddeferr[t, d] ==
#                 sum(m.deferrschedule[t - k, d] * dfdeferrables.loc[d, 'profile_kw'][k]
#                    for k in range(0, min(dfdeferrables.loc[d, 'duration'], t + 1))))

#     # We can only schedule a load once within the time horizon
#     def r_deferrable_schedule_sum(m, d):
#         return (sum(m.deferrschedule[i, d] for i in m.horizon) == 1)

#     # If we are outside of startby - endby, we enforce no operation
#     def r_deferrable_timebounds(m, t, d):
#         if t < dfdeferrables.loc[d, 'startby']:
#             return (m.demanddeferr[t, d] == 0)
#         if t > dfdeferrables.loc[d, 'endby']:
#             return (m.demanddeferr[t, d] == 0)
#         else:
#             return Constraint.Skip

    # --------------------------------------------------------
    # ---------------------Helper-----------------------------
    # --------------------------------------------------------
#     # Useless step which seems to be necessary
#     def r_demand_total(m, t):
#         return (m.demand_controllable[t] ==
#                 sum(m.demandshape[t, s] for s in m.shapeables) +
#                 sum(m.batteryin[t, b] - m.batteryout[t, b] for b in m.batteries) +
#                 sum(m.demanddeferr[t, d] for d in m.deferrables))

    def r_demand_total(m, t):
        return (m.demand_controllable[t] ==
                sum(m.batteryin[t, b] - m.batteryout[t, b] for b in m.batteries))

#     # Limit maximum peak
#     def r_peak_high(m, t):
#         return (m.demand_controllable[t] + demand_uncontrollable[t]
#                 <= m.peakhigh)

#     # Stop constraint at 0
#     def r_peak_high_zero(m, t):
#         return (0 <= m.peakhigh)

#     # Limit minimum peak
#     def r_peak_low(m, t):
#         return (m.peaklow
#                 <= m.demand_controllable[t] + demand_uncontrollable[t])

#     # Stop constraint at 0
#     def r_peak_low_zero(m, t):
#         return (m.peaklow <= 0)

    # Community import (by definition above zero)
    def community_import_zero(m, t):
        return (m.community_import[t] >= 0)

    # Community import (by definition above zero)
    def community_import(m, t):
        return (m.community_import[t] >=
                m.demand_controllable[t] + demand_uncontrollable[t])

#     # Shapeable
#     m.r1 = Constraint(m.horizon, m.shapeables, rule=r_shape_min_power)
#     m.r2 = Constraint(m.horizon, m.shapeables, rule=r_shape_max_power)
#     m.r3 = Constraint(m.shapeables, rule=r_shape_energy)
#     m.r4 = Constraint(m.horizon, m.shapeables, rule=r_shape_timebounds)
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
#     # Deferrable
#     m.r16 = Constraint(m.horizon, m.deferrables, rule=r_deferrable_schedule)
#     m.r17 = Constraint(m.deferrables, rule=r_deferrable_schedule_sum)
#     m.r18 = Constraint(m.horizon, m.deferrables, rule=r_deferrable_timebounds)
    # Helper
    m.r15 = Constraint(m.horizon, rule=r_demand_total)
#     m.r19 = Constraint(m.horizon, rule=r_peak_high)
#     m.r20 = Constraint(m.horizon, rule=r_peak_low)
#     m.r21 = Constraint(m.horizon, rule=r_peak_low_zero)
#     m.r22 = Constraint(m.horizon, rule=r_peak_high_zero)
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
#         else:
#             results = opt.solve(m, tee=verbose)

    if verbose:
        print(results)

    #################################################### Results
    # A Dictionnary contains all the results
    results = {}
#     keys = ['demandshape', 'batteryin',
#             'batteryout', 'batteryenergy',
#             'demanddeferr', 'deferrschedule']
    keys = ['batteryin',
            'batteryout', 'batteryenergy']

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

    # community_import
    results['community_import'] = [ max(0, a + b)
                                  for a, b in zip(
                                  demand_uncontrollable,
                                  results['demand_controllable'])]

#     # High peak
#     results['peakhigh'] = m.peakhigh.get_values()[None]

#     # Low peak
#     results['peaklow'] = m.peaklow.get_values()[None]

    # Total import from the community
    results['total_community_import'] = sum(
        results['community_import'] ) * timestep
    return results
