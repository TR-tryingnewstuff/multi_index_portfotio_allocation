#%%
import pandas as pd
import pandas_ta as ta
import numpy as np
import rllib 
import ray
from ray import tune
from ray import air
import gymnasium
import yfinance as yf
from gymnasium.spaces import Dict, Box
#%%
spx = yf.Ticker('^SPX').history(start='2010-01-01')
spx['symbol'] = ['spx'] * len(spx)
nq = yf.Ticker('^IXIC').history(start='2010-01-01')
nq['symbol'] = ['nq'] * len(nq)
dow = yf.Ticker('^DJI').history(start='2010-01-01')
dow['symbol'] = ['dow'] * len(dow)

def add_features(curr_df):
            curr_df['ema_var'] = curr_df['Close'].pct_change().ewm(span=20).var()
            curr_df['var'] = curr_df['Close'].pct_change().rolling(20).var()  
            curr_df['ema'] = ta.ema(curr_df['Close'],length=20)
            curr_df['macd_w'] = ta.macd(curr_df['Close'],fast=12*5, slow=26*5).values[:, -1]
            curr_df['macd_w_lag'] = ta.macd(curr_df['Close'],fast=12*5, slow=26*5).pct_change().values[:, -1]
            curr_df['macd_m'] = ta.macd(curr_df['Close'],fast=12*22, slow=26*22).values[:, -1]
            curr_df['macd_m_lag'] = ta.macd(curr_df['Close'],fast=12*22, slow=26*22).pct_change().values[:, -1]
            curr_df['range_ema'] = ((curr_df['Close'] - ((curr_df['ema'].rolling(60).min() +curr_df['ema'].rolling(60).max()) /2 ))) / (curr_df['ema'].rolling(60).max() -curr_df['ema'].rolling(60).min())
            curr_df['range'] = ((curr_df['Close'] - ((curr_df['Close'].shift(1).rolling(30).min() +curr_df['Close'].shift(1).rolling(30).max()) /2 ))) / (curr_df['Close'].shift(1).rolling(30).max() -curr_df['Close'].shift(1).rolling(30).min())
            curr_df['zscore'] = ta.zscore(curr_df['Close'])

            return curr_df

spx, nq, dow = add_features(spx), add_features(nq), add_features(dow)


df = pd.concat([spx,nq,dow]).sort_index().dropna()
df.index = pd.to_datetime(df.index.date)
df.drop(['Open', 'High', 'Low','Dividends', 'Stock Splits'], axis=1, inplace=True)
print(df.index)

macro = pd.read_csv('/home/fast-pc-2023/Téléchargements/python/light_gbm_tests-main/macroeconomic_data.csv')
macro.columns = ['date','DFF','DTB4WK','AWHAETP','TCU','UNRATE','STLFSI4','NFCI','VIXCLS','RECPROUSM156N','CFNAIDIFF','DEXUSEU']
macro = macro.set_index('date')

macro = macro.loc[macro['RECPROUSM156N'] < 10]
df = df.loc[df.index.isin(macro.index)]
df


#%%
class Market(gymnasium.Env):
    def __init__(self, env_config):
        self.capital = 10000
        self.bench_pos = np.array([0.25]*4) 
        self.position = np.array([0.25]*4)

        self.action_space = Box(0,1,shape=(4,))

        self.date =  '2012-04-23'
        self.next_date = '2012-04-24'
        self.get_obs()

        self.observation_space = Dict({
            'index': Box(-np.inf, np.inf, shape=self.index_data.shape), 
            'macro': Box(-np.inf, np.inf, shape=self.macro_data.shape), 
            'position': Box(0, 1, self.position.shape)})
        


    def reset(self, seed=123, options={}):

        self.capital = 10000
        self.position = np.array([0.25]*4) 

        # ? try implementing some sort of optimal sampling
        #self.date = df.sample(n=1).index
        self.next_date = str(df.iloc[df.index.get_loc(self.date).stop + 1, :].name.date())

        if self.index_data.index.year[0] == 2023:
                self.date =  str(pd.to_datetime('2012-04-23').date())
                self.next_date = str(pd.to_datetime('2012-04-24').date())


        return self.obs
    
    def step(self, action):
        self.next_date = str(df.iloc[df.index.get_loc(self.date).stop + 1, :].name.date())


        self.get_reward(action)
        self.get_obs()

        Done = False
        if self.index_data.index.weekday[0] == 1:
            Done = True
        
        return self.obs, self.reward.sum(), Done, {}
    
    def get_reward(self, action):
        commissions = abs(self.position - action).sum() * self.capital * 0.002
        self.position = action
        print(action)
        #print(self.index_data, '\n' ,df.loc[self.next_date].sort_values('symbol')['Close'])
        change = (self.index_data['Close'].values - df.loc[self.next_date].sort_values('symbol')['Close'].values) / self.index_data['Close'].values 
        change = np.hstack([change, np.array(0)])

      
        self.reward = (self.position * change - self.bench_pos * change) * self.capital
       
    
    def get_obs(self):

        self.date = self.next_date
        self.macro_data = macro.loc[self.date]
        self.index_data = df.loc[self.date].sort_values('symbol').drop('symbol', axis=1)
        #print(self.index_data.values, self.index_data.dtypes)

        self.obs = {'index': self.index_data.values,
                    'macro': self.macro_data.values,
                    'position': self.position}
        
#env = Market({})




#for i in range(10):
 #   obs, reward, done, info = env.step(env.action_space.sample())

 #   print(obs, '\n', reward)

# %%

from ray.rllib.algorithms.es import ESConfig
from custom_model import KerasModel
from ray.rllib.models.catalog import ModelCatalog
ModelCatalog.register_custom_model("keras_model", KerasModel)

config = ESConfig()
config = config.training(eval_prob=0.4, report_length=10,model={'custom_model': 'keras_model'}, episodes_per_batch=1)

config = config.environment(Market).rollouts(num_rollout_workers=20).framework('tf')



tune.Tuner(  

    "ES",
    
    run_config=ray.train.RunConfig(),
    tune_config=tune.TuneConfig(
            mode='max'),
    param_space=config.to_dict(),

).fit()

# %%
