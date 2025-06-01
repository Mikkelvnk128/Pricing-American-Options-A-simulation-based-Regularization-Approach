import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import itertools
import matplotlib.pyplot as plt

# Own import
from price_optimizer import american_price
from EOP import EOP

# Initial configuration
sim_params = {
    'S0':       100, 
    'T':        1.0, 
    'vol':      0.2, 
    'r':        0.0,
    'q':        0.0,
    'num_paths':10000,
    'time_steps':10
}

opt_params = {
    'strike': 100,
    'style':  'maxcall'
}

reg_params = {
    'num_basis':      3, 
    'method':        'RLSM', 
    'include_payoff': True,
    'p':              None # To be optimized for RLSM and DRLSM
}

heston_cfg = {'v0':0.01,'kappa': 2.0,'theta':0.01,'sigma': 0.2,'rho':-0.3} 




# Setup to be tested
S0_list    = [80,100,120]
und_list   = [5,10,50,100]
methods    = ['LSM','DLSM','RLSM','DRLSM']

num_runs = 10
base_seed = 1
seeds = [base_seed + i for i in range(num_runs)] # Ensure different seeds for every run but same for every method

total_combos = len(list(itertools.product(methods,S0_list,und_list)))

step = 0 # step counter'

executor = Parallel(n_jobs=-1) # Parallel executor


def sim(model):
    
    step = 0

    # Run the test
    records = []
    for method in methods:
        # pick basis size per method
        nb = 10 if method in ('RLSM','DRLSM') else 3
    
        for S0 in S0_list:
                for d in und_list:
    
                    # Override paramters
                    sp               = sim_params.copy()
                    sp['S0']         = [S0]*d
                    rp               = reg_params.copy()
                    rp['method']     = method
                    rp['num_basis']  = nb
    
                    t0 = time.time()
                    sims = executor(delayed(american_price)(sp,opt_params,rp,model,heston_cfg,seed,True)
                        for seed in seeds)
                    elapsed = time.time()-t0
    
                    ins  = np.array([x[0] for x in sims])
                    outs = np.array([x[1] for x in sims])
                    
                    eops = Parallel(n_jobs=-1)(delayed(EOP)({**sp,'num_paths': 2**16},opt_params,seed,model,heston_cfg) for seed in seeds)
    
                    records.append({
                        'Method':       method,
                        'S0':           S0,
                        'd':            d,
                        'Mean_In':      ins.mean(),
                        'Mean_Out':     outs.mean(),
                        'SD_In':        ins.std(ddof=1),
                        'SD_Out':       outs.std(ddof=1),
                        'Run_Time_sec': elapsed,
                        'EOP':          np.mean(eops),
                        'EOP std':      np.std(eops,ddof=1)
                        
                    })
                    
                    step += 1
    
                    print(f"Step {step} out of {total_combos} completed.")
    
    # The table
    df   = pd.DataFrame(records)
    df.sort_values(by=['Method','d','S0'],inplace=True)
    
    df['Method'] = pd.Categorical(df['Method'],categories=methods,ordered=True)
    df.sort_values(by=['Method','d','S0'], inplace=True)

    df['rel_err_in_bp']  = 10000*(df['Mean_In']-df['EOP'])/df['EOP']
    df['rel_err_out_bp'] = 10000*(df['Mean_Out']-df['EOP'])/df['EOP']

    df['Scenario'] = df.groupby(['d','S0'],sort=False).ngroup()+1

    return df

df_bs = sim('Black-Scholes')
df_h  = sim('Heston')


methods = ['LSM','DLSM','RLSM','DRLSM']

fontsize = 13
plt.figure(figsize=(10,10))
plt.grid(alpha=0.3)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for m,col in zip(methods,colors):
    sub = df_bs[df_bs['Method']==m]
    # out-of-sample curve
    plt.plot(sub['Scenario'], sub['rel_err_out_bp'],ms=4,label=f'{m} Out',color=col)
    # in-sample mean
    plt.hlines(sub['rel_err_in_bp'].mean(),1,12,colors=col,linestyles='--',linewidth=1,label=f'{m} In mean')
    # out-of-sample mean
    plt.hlines(sub['rel_err_out_bp'].mean(),1,12,colors=col,linestyles='-',linewidth=1,label=f'{m} Out mean')

plt.axhline(0,color='grey',lw=1)
plt.xticks(range(1,13),fontsize=fontsize)
plt.yticks(fontsize=fontsize) 
plt.xlabel('Scenario',fontsize=fontsize)
plt.ylabel('Relative Error in basis points',fontsize=fontsize)
plt.legend(fontsize=fontsize, ncol=2, loc='lower left')
plt.tight_layout()
ymin, ymax = plt.gca().get_ylim()
plt.show()


means = df_bs.groupby('Method')[['rel_err_in_bp', 'rel_err_out_bp']].mean()
print(means)


plt.figure(figsize=(10,10))
plt.grid(alpha=0.3)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for m,col in zip(methods,colors):
    sub = df_h[df_h['Method']==m]
    # out-of-sample curve
    plt.plot(sub['Scenario'], sub['rel_err_out_bp'],ms=4,label=f'{m} Out',color=col)
    # in-sample mean
    plt.hlines(sub['rel_err_in_bp'].mean(),1,12,colors=col,linestyles='--',linewidth=1,label=f'{m} In mean')
    # out-of-sample mean
    plt.hlines(sub['rel_err_out_bp'].mean(),1,12,colors=col,linestyles='-',linewidth=1,label=f'{m} Out mean')

plt.axhline(0,color='grey',lw=1)
plt.xticks(range(1,13), fontsize=fontsize)
plt.yticks(fontsize=fontsize)   
plt.xlabel('Scenario',fontsize=fontsize)
plt.ylabel('Relative Error in basis points',fontsize=fontsize)
plt.legend(fontsize=fontsize, ncol=2, loc='lower left')
plt.tight_layout()
plt.show()


means = df_h.groupby('Method')[['rel_err_in_bp', 'rel_err_out_bp']].mean()
print(means)







