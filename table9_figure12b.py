import itertools
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Own imports
from upper_bound import dual_upper_bound
from price_optimizer import american_price
#from CRR import CRR_put

# Initial configuration
sim_params = {
    'S0':       36, 
    'T':        1.0, 
    'vol':      0.2, 
    'r':        0.06,
    'q':        0.0,
    'num_paths':2**18,
    'time_steps':50
}

opt_params = {
    'strike': 40,
    'style':  'put'
}

reg_params = {
    'num_basis':      3, 
    'method':        'RLSM', 
    'include_payoff': False,
    'p':              None # To be optimized for RLSM and DRLSM
}

heston_cfg = {'v0': 0.01,'kappa': 2.0,'theta': 0.01,'sigma': 0.2,'rho':   -0.3} 

model = 'Heston'

L_out = 2**11
L_in = 500

# Setup to be tested
S0_list    = [36,38,40,42,44]
T_list     = [1,2]
vol_list   = [0.2,0.4]
methods    = ['LSM','DLSM','RLSM','DRLSM']

seed = 123

total_combos = len(list(itertools.product(methods,S0_list,T_list,vol_list)))

step = 0 # step counter'

# Run the test
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
                hc               = heston_cfg.copy()
                hc['sigma']      = vol 
                rp               = reg_params.copy()
                rp['method']     = method
                rp['num_basis']  = nb

                t0 = time.time()
                
                in_p,out_p,beta_dict = american_price(sp,opt_params,rp,model,hc,seed,out_of_sample= True)
                dual_gap = dual_upper_bound(sp,opt_params,beta_dict,rp,model,hc,L_out,L_in,seed+100)
                
                elapsed = time.time()-t0

                records.append({
                    'Method':       method,
                    'S0':           S0,
                    'T':            T,
                    'Vol':          vol,
                    'Price_In':     in_p,
                    'Price_Out':    out_p,
                    'Dual':         dual_gap,
                    'Upper_Bound':  out_p+dual_gap,  
                    'Run_Time_sec': elapsed
                })
                
                step += 1
                
                print(f"Step {step} out of {total_combos} completed.")

# The table
df   = pd.DataFrame(records)

df.sort_values(by=['Method','S0','Vol','T'], inplace=True)

print(df)



df['Scenario'] = df.groupby(['S0','T','Vol'], sort=False).ngroup() + 1


sc_mean_ub = df.groupby('Scenario')['Upper_Bound'].transform('mean')
df['rel_err_ub_bp'] = 10000 * (df['Upper_Bound']-sc_mean_ub)/sc_mean_ub

methods  = ['LSM','DLSM','RLSM','DRLSM']
fontsize = 13

plt.figure(figsize=(10,8))
plt.grid(alpha=0.3)

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
max_scen = df['Scenario'].max()

for m, col in zip(methods, colors):
    sub = df[df['Method'] == m]
    x = sub['Scenario']
    y = sub['rel_err_ub_bp']

    plt.plot(x,y,ms=4,label=f'{m}',color=col)
    plt.hlines(y.mean(),1,max_scen,colors=col,linestyles='--',linewidth=1,label=f'{m} mean')

plt.axhline(0, color='grey', lw=1)
plt.xticks(np.arange(1, max_scen + 1), fontsize=fontsize)
plt.xlabel('Scenario', fontsize=fontsize)
plt.ylabel('Performance Devation in basis points', fontsize=fontsize)
plt.yticks(fontsize=fontsize)
plt.legend(fontsize='large',ncol=2,loc='lower left')
plt.tight_layout()
plt.show()



means = df.groupby('Method')['rel_err_ub_bp'].mean()
print(means)










