import streamlit as st
import pandas as pd
import talib as ta
import numpy as np
import random
import torch


from NepseData import stock_dataFrame
from ppo import PPOAgent
from tradingenv import TradingEnvGeneral2



seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

actions = {0:"Hold",1:"Buy",2:"Sell"}


def main():
    print("NumPy version:", np.__version__)
    st.title("PPO for stock trading")

    #Ask user to enter the symbol listed in NEPSE
    stock = st.text_input('Enter Company name')

    if (stock != ""):
        try:
            df = stock_dataFrame(stock)
        except:
            print("An exception occured")
        
    
        st.subheader("Head of Data")
        st.dataframe(df.head())

        df.interpolate(method='linear', inplace=True)
        df['CCI']= ta.CCI(df['High'],df['Low'],df['Close'])
        df['WCLPrice'] = ta.WCLPRICE(df['High'],df['Low'],df['Close'])
        df['SAREXT'] = ta.SAREXT(df['High'],df['Low'])
        _,df['Slowd'] = ta.STOCH(df['High'],df['Low'],df['Close'])
        df['TypePrice'] = ta.TYPPRICE(df['High'],df['Low'],df['Close'])
        df['ADX'] = ta.ADX(df['High'],df['Low'],df['Close'],timeperiod=14)
        df['RSI'] = ta.RSI(df['Close'],timeperiod=14)
        df['EMA']= ta.EMA(df['Close'], timeperiod=30)
        df['MACD'],_,__ = ta.MACD(df['Close'],fastperiod=12,slowperiod=26,signalperiod=9)
        df['OBV'] = ta.OBV(df['Close'],df['Volume'])
        techindicators = ['CCI','WCLPrice','SAREXT','Slowd','TypePrice']
        df3 = df.copy()
        df3.dropna(inplace=True)
        selected_columns = techindicators + ['Close']
        df3 = df3[selected_columns]

        env = TradingEnvGeneral2(df3,indicators=techindicators,time_period=20)
        env.seed(seed)
        action = env.action_space.sample()  # Take a random action
        obs, reward, done, info = env.step(action)
        print(obs)
        print(reward)
        print(done)
        print(info)



        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        env.seed(seed)
        print(state_dim)
        buffer_size = 5000
        gamma = 0.99
        K_epochs = 4
        eps_clip = 0.2

        # Create the agent
        agent = PPOAgent(state_dim, action_dim, buffer_size, 
                        gamma, K_epochs, eps_clip,hidden_dim=128)

        # Train the agent
        agent.train(env,num_episodes=20)
        st.write("Training done")
        plot = agent.plot()
        st.pyplot(plot)

        df4 = df3[-20:]
        env = TradingEnvGeneral2(df4,indicators=techindicators,time_period=20)
        state = env.reset()
        done = False
        while not done:
            
            state = agent.normalize_state(state)
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            dist = agent.policy_old.pi(state_tensor)
            action = dist.sample()
            state, reward, done, _ = env.step(action.item())
            state = agent.normalize_state(state)
           
        st.write(actions[action.item()])



        st.write(df3[-20:])


if __name__ == '__main__':
    main()