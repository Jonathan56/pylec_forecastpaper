"""
Functions to validate result (i.e., simple tests)
"""

def battery(df, energy_bounds=(0, 102),
            power_bounds=(-34, 34), end_energy=51):
    """
    Dataframe with the controllable demand from the optimization.
    """
    # Energy (end_energy == init_energy)
    e = (end_energy
         + df.f_battery_kW.apply(lambda x: 0 if x < 0 else x).cumsum()
         * 0.95 * 15 / 60
         - df.f_battery_kW.apply(lambda x: 0 if x > 0 else x).abs().cumsum()
         / 0.95 * 15 / 60)

    # Energy constraints
    assert e.min() >= energy_bounds[0] - 1  # Margin
    assert e.max() <= energy_bounds[1] + 1  # Margin

    # Power constraints
    assert df['f_battery_kW'].min() >= power_bounds[0]
    assert df['f_battery_kW'].max() <= power_bounds[1]
    return e.min().round(2), e.max().round(2)
