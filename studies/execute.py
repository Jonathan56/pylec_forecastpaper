from fppylec import util  # <-- CHANGE NAME
import papermill as pm
import pickle
from datetime import datetime
import os
import sys
import pytest

container = sys.argv[1]
name =      sys.argv[2].split('/')[1]  # to remove the studies/...
pwd =       sys.argv[3]
output = name.split('.')[0] + '_output'

# # Run pytest
# test_r = pytest.main(['../tests/'])

# Send an email simulation is starting
util.send_email(f'{name} started on {container}',
                f'{name} started at {datetime.now()} on container {container}.', _pwd=pwd)

# Launch notebook through papermill
try:
    nb = pm.execute_notebook(
            name,  # --> potentially send email from the notebook
            output + '.ipynb',
            kernel_name='python',
            log_output=True,
            stdout_file=sys.stdout,
            stderr_file=sys.stderr,
    )
except:
    # Keep on going
    pass

# Turn notebook to html
os.system(f'jupyter nbconvert {output} --to html --output {output + ".html"}')

# Read result object
try:
    with open('result.pickle', 'rb') as r_file:
        df = pickle.load(r_file)
    # Send simulation results via email
    util.send_email(f'{name} succeeded on {container}',
                    f'{name} succeeded at {datetime.now()}',
                    df, html=output + '.html', _pwd=pwd)
    # Remove temporary files
    os.system(f'rm {output + ".ipynb"} {output + ".html"} result.pickle')

except:
    # Send failure via email
    util.send_email(f'{name} failed on {container}',
                    f'{name} failed at {datetime.now()}',
                    html=output + '.html', _pwd=pwd)
    # Remove temporary files
    os.system(f'rm {output + ".ipynb"} {output + ".html"}')
# PS: tabulate(df, headers='keys', tablefmt='psql') looks good !
