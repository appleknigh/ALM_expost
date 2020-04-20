

#%%
def res_func(x):
    return ALM_kit.optimize_duration(
        x,
        df_PVCF['cf_asset'],
        df_PVCF['t_asset'],
        df_forcast['fit_par'],
        df_PVCF['LDur'])

#%%
asset_text = '0 1 0'
asset_value = np.fromstring(asset_text, dtype=float, sep=' ')
res = minimize(res_func, asset_value, method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True})

#%%
asset_value = ' '.join(res.x.astype(str))
AssetDisable_status = True

#%%
def fun_loss(x):
    res_loss = utility.PVCashflow_AL(df_forcast, bond_weight=abs(x),N=1)
    return abs(res_loss['ADur']-res_loss['LDur'])

bnds = ((0,1),(0,1),(0,1))

res = minimize(res_fun2, asset_value,
               method='nelder-mead',
               options={'xatol': 1e-8, 'disp': True})

print(res_fun2(abs(res.x)))

#%%
def fun_loss(x):
    res_loss = utility.PVCashflow_AL(df_forcast, bond_weight=abs(x),N=1)
    return abs(res_loss['ADur']-res_loss['LDur'])

res = minimize(fun_loss,asset_value,bounds=bnds)
x = res.x/np.sum(res.x)
print(res_fun2(x))

# %%
# Asset mix
def fun_loss(x):
    res_loss = utility.PVCashflow_AL(df_forcast, bond_weight=abs(x),N=1)
    return abs(res_loss['ADur']-res_loss['LDur'])

def opt_dur(asset_value,bnds):
    res = minimize(fun_loss, asset_value,bounds=bnds)        
    asset_value_norm = np.round(res.x/np.sum(res.x),3)
    asset_text = ' '.join(asset_value_norm.astype(str))
    return asset_text

bnds = ((0,1),(0,1),(0,1))

asset_value = np.array([0.5, 0.5, 0])
asset_text = opt_dur(asset_value,bnds)
print(asset_value)
print(asset_text)

# %%
