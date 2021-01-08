"""
Cost allocation functions
"""
from tqdm.notebook import tqdm
from pylec import metric
import pytest
import numpy as np
import pandas


def sharing_rule(cake, coefs, indconso):
    # There is a surplus of energy so everyone gets what he consumed
    assert cake <= indconso.sum() + 1e-2, f'Shareable cake = {cake}, conso passive = {indconso.sum()}'

    # Sharing must be adjusted among participants
    shares = {column: 0 for column in indconso.index}
    current_cake = cake

    # Split cake with coefs among consummers
    count = 0
    while current_cake > 1e-4 or count < 100:
        for name in shares.keys():
            shares[name] = min(indconso[name], shares[name] + current_cake * coefs[name])
        current_cake = cake - sum(shares.values())
        assert current_cake >= -1e-3, f'current_cake = {current_cake}'
        count += 1
    assert pytest.approx(sum(shares.values()), abs=1e-2) == cake, f'difference = {sum(shares.values()) - cake}'
    return shares

def sharing_rule_faster(cake, coefs, indconso):
    # There is a surplus of energy so everyone gets what he consumed
    assert cake <= indconso.sum() + 1e-6, f'Shareable cake = {cake}, conso passive = {indconso.sum()}'

    # Sharing must be adjusted among participants
    shares = {column: 0 for column in indconso.index}
    current_cake = cake

    # Split cake with coefs among consummers
    next_iteration_p = list(shares.keys())
    next_iteration_c = coefs.sum()
    assert next_iteration_c == pytest.approx(1, abs=0.001), 'coefs add up to one'
    while len(next_iteration_p) != 0 and current_cake > 1e-4:
        participants = list(next_iteration_p)
        total_coef = float(next_iteration_c)
        #print(f'Total coef {total_coef}')
        for name in participants:
            shares[name] = min(indconso[name], shares[name] + current_cake * coefs[name]/total_coef)
            #print(f'name {name}, share {shares[name]}, coef {coefs[name]/total_coef}')
            if shares[name] == indconso[name]:
                next_iteration_p.remove(name)
                next_iteration_c -= coefs[name]
        current_cake = cake - sum(shares.values())
        assert current_cake >= -1e-3, f'current_cake = {current_cake}'
    assert pytest.approx(sum(shares.values()), abs=1e-4) == cake, f'difference = {sum(shares.values()) - cake}'
    return shares

def marginal(df, consumption, storage, production, horizon, indconso_file, coefs=False, verbose=True):
    """
    """
    md = df[[consumption, storage, production]].copy()
    assert not md.isnull().values.any(), 'Include NaN values'
    md_freq = int(pandas.infer_freq(md.index)[:-1])
    assert md_freq == horizon, 'Warning check that it makes sense here'

    # What's the cake to share?
    localcons = (md[[consumption, production]].min(axis=1)  # passive local conso
                 + ((md[consumption] - md[production]).clip(lower=0) # diff between positive net load w/o discharing
                    - (md[consumption] - md[production] + md[consumption].clip(upper=0)).clip(lower=0)))
    localcons = localcons.to_frame('passive')
    assert any(localcons['passive'] >= 0)
    trad = metric.localconsumption(md, consumption, storage, production, horizon, timeseries=True)
    assert trad.sum() >= localcons['passive'].sum(), ('Passive self-conso + battery discharge' +
                                                      'should be less than traditional self-conso')

    # What are individual consumption
    indconso = pandas.read_csv(indconso_file, parse_dates=[0], index_col=[0])
    indconso = indconso.drop(['vo_houses_kW', 'vo_pv_coef'], axis=1)
    if verbose:
        print(f'Participants: {indconso.columns}')

    # Allocation coeficients are set to be equal by default
    if not coefs:
        coefs = indconso.iloc[0].copy()
        coefs.loc[:] = 1 / len(coefs)

    # Run allocation mechanism
    indexes = []
    allocation = []
    for index, row in tqdm(localcons.iterrows(), total=len(localcons), desc='Progress:', miniters=5000):
        if row['passive'] > 0:
            tmp = sharing_rule_faster(
                cake=row['passive'],
                coefs=coefs,
                indconso=indconso.loc[row.name, :])
            assert pytest.approx(sum(tmp.values()), abs=1e-4) == row['passive']
            indexes.append(row.name)
            allocation.append(tmp.copy())
        else:
            indexes.append(row.name)
            tmp = coefs.copy()
            tmp.loc[:] = 0
            allocation.append(tmp.to_dict().copy())
    allocations = pandas.DataFrame(index=indexes, data=allocation)

    # Compare to expected results
    ratio_actual_expected = []
    for column in allocations.columns:
        ratio_actual_expected.append(
            allocations.loc[:, column].sum() /
            (localcons['passive'].sum() * coefs[column]))
    assert np.mean(ratio_actual_expected) == pytest.approx(1, abs=0.01), (f'Moyenne de la répartition' + f'{np.mean(ratio_actual_expected)}')

    # Return results
    return {'allocations': allocations,
            'actual_expected': pandas.DataFrame(index=allocations.columns,
                                                data={'ratio': ratio_actual_expected})}
