import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import itertools
import matplotlib.pyplot as plt

# Own import
from price_optimizer import american_price


# Initial configuration
sim_params = {
    'S0':       36, 
    'T':        1, 
    'vol':      0.2, 
    'r':        0.06,
    'q':        0.0,
    'num_paths':1024,
    'time_steps':50
}

opt_params = {
    'strike': 40,
    'style':  'put'
}

reg_params = {
    'num_basis':      3, 
    'method':        'RLSM', 
    'include_payoff': True,
    'p':              None # To be optimized for RLSM and DRLSM
}

heston_cfg = {'v0':0.01,'kappa':2.0,'theta':0.01,'sigma':0.2,'rho':-0.3} 

model = 'Heston'


# Setup to be tested
methods    = ['LSM','DLSM','RLSM','DRLSM']
num_paths_range = [2**8,2**9,2**10,2**13,2**16]

num_runs = 100
base_seed = 1
seeds = [base_seed + i for i in range(num_runs)] # Ensure different seeds for every run but same for every method

total_combos = len(list(itertools.product(methods,num_paths_range)))

step = 0 # step counter'

executor = Parallel(n_jobs=-1) # Parallel executor

# Run the test

records = []
for method in methods:
    # pick basis size per method
    nb = 10 if method in ('RLSM','DRLSM') else 3

    for paths in num_paths_range:

                # Override paramters
                sp               = sim_params.copy()
                sp['num_paths']  = paths 
                rp               = reg_params.copy()
                rp['method']     = method
                rp['num_basis']  = nb

                t0 = time.time()
                sims = executor(delayed(american_price)(sp,opt_params,rp,model,heston_cfg,seed,True)
                    for seed in seeds)
                elapsed = time.time()-t0

                ins  = np.array([x[0] for x in sims])
                outs = np.array([x[1] for x in sims])

                records.append({
                    'Method':       method,
                    'num_paths':    paths,
                    'Mean_In':      ins.mean(),
                    'Mean_Out':     outs.mean(),
                    'SD_In':        ins.std(ddof=1),
                    'SD_Out':       outs.std(ddof=1),
                    'Run_Time_sec': elapsed
                })
                
                step += 1
                
                print(f"Step {step} out of {total_combos} completed.")

# The table
df_1   = pd.DataFrame(records)
df_1.sort_values(by=['Method','num_paths'], inplace=True)
df_1['Scenario'] = df_1.groupby(['num_paths'],sort=False).ngroup()+1
methods = ['LSM','DLSM','RLSM','DRLSM']

step = 0
records = []
for method in methods:
    # pick basis size per method
    nb = 10 if method in ('RLSM','DRLSM') else 3
    step = 0
    for paths in num_paths_range:

                # Override paramters
                sp               = sim_params.copy()
                sp['num_paths']  = paths 
                sp['T']          = 2 
                sp['S0']         = 44 
                rp               = reg_params.copy()
                rp['method']     = method
                rp['num_basis']  = nb
                h_cfg            = heston_cfg.copy()
                h_cfg['sigma']   = 0.4
                

                t0 = time.time()
                sims = executor(delayed(american_price)(sp,opt_params,rp,model,h_cfg,seed,True)
                    for seed in seeds)
                elapsed = time.time()-t0

                ins  = np.array([x[0] for x in sims])
                outs = np.array([x[1] for x in sims])

                records.append({
                    'Method':       method,
                    'num_paths':    paths,
                    'Mean_In':      ins.mean(),
                    'Mean_Out':     outs.mean(),
                    'SD_In':        ins.std(ddof=1),
                    'SD_Out':       outs.std(ddof=1),
                    'Run_Time_sec': elapsed
                })
                
                step += 1
                
                print(f"Step {step} out of {total_combos} completed.")

# The table
df_20   = pd.DataFrame(records)
df_20.sort_values(by=['Method','num_paths'], inplace=True)
df_20['Scenario'] = df_20.groupby(['num_paths'],sort=False).ngroup()+1



plt.figure(figsize=(6,4))
plt.grid(alpha=0.3)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for m,col in zip(methods,colors):
    sub = df_1[df_1['Method']==m]
    sub = sub.reset_index(drop=True)
    x   = range(len(sub))
    # out-of-sample 
    plt.errorbar(x, sub['Mean_Out'],yerr=sub['SD_Out']/np.sqrt(num_runs),fmt='-',color=col,label=f'{m}',capsize=3)

plt.xticks(ticks=x,labels=[f'$2^{{{int(np.log2(n))}}}$' for n in sub['num_paths']])
plt.axhline(0,color='grey',lw=1)
plt.xlabel('Number of paths')
plt.ylim(3.85,3.98)
plt.ylabel('Price')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()



plt.figure(figsize=(6,4))
plt.grid(alpha=0.3)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for m,col in zip(methods,colors):
    sub = df_20[df_20['Method']==m]
    sub = sub.reset_index(drop=True)
    x   = range(len(sub))
    # out-of-sample 
    plt.errorbar(x, sub['Mean_Out'],yerr=sub['SD_Out']/np.sqrt(num_runs),fmt='-',color=col,label=f'{m}',capsize=3)

plt.xticks(ticks=x,labels=[f'$2^{{{int(np.log2(n))}}}$' for n in sub['num_paths']])
plt.axhline(0,color='grey',lw=1)
plt.xlabel('Number of paths')
plt.ylim(0.5,0.75)
plt.ylabel('Price')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

