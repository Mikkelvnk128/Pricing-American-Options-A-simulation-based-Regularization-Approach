import time
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import itertools
import matplotlib.pyplot as plt

# Own import
from price_optimizer import american_price
from CRR import CRR_put


# Initial configuration
sim_params = {
    'S0':       36, 
    'T':        1.0, 
    'vol':      0.2, 
    'r':        0.06,
    'q':        0.0,
    'num_paths':2**13,
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
S0_list    = [36,38,40,42,44]
T_list     = [1,2]
vol_list   = [0.2,0.4]
methods    = ['LSM','DLSM','RLSM','DRLSM']

num_runs = 100
base_seed = 1
seeds = [base_seed + i for i in range(num_runs)] # Ensure different seeds for every run but same for every method

total_combos = len(list(itertools.product(methods,S0_list,T_list,vol_list)))

step = 0 # step counter'

executor = Parallel(n_jobs=-1) # Parallel executor

def sim(paths):
    
    step = 0

    # Run the test for 2^10
    records = []
    for method in methods:
        # pick basis size per method
        nb = 10 if method in ('RLSM','DRLSM') else 3
    
        for S0 in S0_list:
            for T in T_list:
                for vol in vol_list:
    
                    # Override paramters
                    sp               = sim_params.copy()
                    sp['S0']         = S0
                    sp['T']          = T
                    sp['vol']        = vol
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
                        'S0':           S0,
                        'T':            T,
                        'Vol':          vol,
                        'Mean_In':      ins.mean(),
                        'Mean_Out':     outs.mean(),
                        'SD_In':        ins.std(ddof=1),
                        'SD_Out':       outs.std(ddof=1),
                        'Run_Time_sec': elapsed
                    })
                    
                    step += 1
                    
                    print(f"Step {step} out of {total_combos} completed.")
    
    # The table
    df   = pd.DataFrame(records)
    
    # Append the price given by CRR for comparison
    n = 2000

    df['CRR'] = df.apply(lambda x: CRR_put(x['S0'],x['T'],opt_params['strike'],sim_params['r'],x['Vol'],n*x['T'],sim_params['time_steps']),axis=1)
    df['Method'] = pd.Categorical(df['Method'],categories=methods,ordered=True)
    df.sort_values(by=['Method','S0','Vol','T'], inplace=True)

    df['Scenario'] = df.groupby(['S0','T','Vol'],sort=False).ngroup()+1
    
    sc_mean_in  = df.groupby('Scenario')['Mean_In'] .transform('mean')
    sc_mean_out = df.groupby('Scenario')['Mean_Out'].transform('mean')

    df['rel_err_in_bp']  = 10000*(df['Mean_In']-sc_mean_in)/sc_mean_in
    df['rel_err_out_bp'] = 10000*(df['Mean_Out']-sc_mean_out)/sc_mean_out
    
    return df


df_10 = sim(2**10)
print(df_10)

df_13 = sim(2**13)
print(df_13)
    

methods = ['LSM','DLSM','RLSM','DRLSM']

fontsize = 13
plt.figure(figsize=(10,10))
plt.grid(alpha=0.3)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for m,col in zip(methods,colors):
    sub = df_10[df_10['Method']==m]
    # out-of-sample curve
    plt.plot(sub['Scenario'], sub['rel_err_out_bp'],ms=4,label=f'{m} Out',color=col)
    # in-sample mean
    plt.hlines(sub['rel_err_in_bp'].mean(),1,20,colors=col,linestyles='--',linewidth=1,label=f'{m} In mean')
    # out-of-sample mean
    plt.hlines(sub['rel_err_out_bp'].mean(),1,20,colors=col,linestyles='-',linewidth=1,label=f'{m} Out mean')

plt.axhline(0,color='grey',lw=1)
plt.xticks(range(1,21),fontsize=fontsize)
plt.yticks(fontsize=fontsize) 
plt.xlabel('Scenario',fontsize=fontsize)
plt.ylabel('Performance Deviation in basis points',fontsize=fontsize)
plt.legend(fontsize=fontsize, ncol=2, loc='lower left')
plt.tight_layout()
ymin, ymax = plt.gca().get_ylim()
plt.show()


means = df_10.groupby('Method')[['rel_err_in_bp', 'rel_err_out_bp']].mean()
print(means)




plt.figure(figsize=(10,10))
plt.grid(alpha=0.3)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for m,col in zip(methods,colors):
    sub = df_13[df_13['Method']==m]
    # out-of-sample curve
    plt.plot(sub['Scenario'], sub['rel_err_out_bp'],ms=4,label=f'{m} Out',color=col)
    # in-sample mean
    plt.hlines(sub['rel_err_in_bp'].mean(),1,20,colors=col,linestyles='--',linewidth=1,label=f'{m} In mean')
    # out-of-sample mean
    plt.hlines(sub['rel_err_out_bp'].mean(),1,20,colors=col,linestyles='-',linewidth=1,label=f'{m} Out mean')

plt.axhline(0,color='grey',lw=1)
plt.xticks(range(1,21), fontsize=fontsize)
plt.yticks(fontsize=fontsize)   
plt.xlabel('Scenario',fontsize=fontsize)
plt.ylabel('Performance Deviation in basis points',fontsize=fontsize)
#plt.ylim(ymin, ymax)
plt.legend(fontsize=fontsize, ncol=2, loc='lower left')
plt.tight_layout()
plt.show()


means = df_13.groupby('Method')[['rel_err_in_bp', 'rel_err_out_bp']].mean()
print(means)

