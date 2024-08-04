import gym
from gym import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import gym
from gym import spaces
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class TradingEnvFiveIndicators(gym.Env):
    """
    A custom trading environment for reinforcement learning in OpenAI's Gym framework.
    
    This environment simulates trading using a given historical price data (in DataFrame format)
    and allows the agent to take actions to buy, sell, or hold the asset.
    
    Attributes:
    -----------
    df : DataFrame
        The historical price data containing 'CCI', 'WCLPrice', 'SAREXT', 'Slowd', 'TypePrice', and 'Close' columns.
    current_step : int
        The current time step in the simulation.
    balance : float
        The current balance in the agent's account.
    shares_held : int
        The number of shares currently held by the agent.
    net_worth : float
        The net worth of the agent's account (balance + value of shares held).
    max_net_worth : float
        The maximum net worth achieved during the simulation.
    initial_balance : float
        The initial balance at the start of the simulation.
    transaction_cost : float
        The cost of making a transaction (e.g., 0.1% of the trade value).
    initial_price : float
        The initial closing price at the start of the simulation.
    returns : list
        A list to track daily returns for calculating the Sharpe ratio.
    scaler : MinMaxScaler
        Scaler for normalizing the observation data.
    """
    
    def __init__(self, df):
        """
        Initialize the trading environment with the given DataFrame of historical price data.

        Parameters:
        -----------
        df : DataFrame
            The historical price data containing 'CCI', 'WCLPrice', 'SAREXT', 'Slowd', 'TypePrice', and 'Close' columns.
        """
        self.df = df
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_net_worth = self.balance
        self.initial_balance = self.balance

        # Action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: values of the specified columns (excluding 'Close')
        self.observation_space = spaces.Box(low=0, high=1, shape=(5,), dtype=np.float32)

        # Example transaction cost: 0.1% per trade
        self.transaction_cost = 0.001
        self.initial_price = self.df.iloc[0]['Close']
        self.returns = []

        # Normalizing the observation data (excluding 'Close')
        self.scaler = StandardScaler()
        self.df[['CCI', 'WCLPrice', 'SAREXT', 'Slowd', 'TypePrice']] = self.scaler.fit_transform(self.df[['CCI', 'WCLPrice', 'SAREXT', 'Slowd', 'TypePrice']])

    def reset(self):
        """
        Reset the environment to the initial state and return the initial observation.
        
        Returns:
        --------
        np.ndarray
            The initial observation of the environment (values of the specified columns).
        """
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_net_worth = self.balance
        self.returns = []
        return self._next_observation()
    
    def _next_observation(self):
        """
        Get the next observation, which consists of the normalized values of the specified columns.
        
        Returns:
        --------
        np.ndarray
            The normalized values of the specified columns.
        """
        return self.df.iloc[self.current_step][['CCI', 'WCLPrice', 'SAREXT', 'Slowd', 'TypePrice']].values
    
    def _take_action(self, action):
        """
        Execute the given action (buy, sell, hold) and update the environment state accordingly.
        
        Parameters:
        -----------
        action : int
            The action to take (0: Hold, 1: Buy, 2: Sell).
        """
        current_price = self.df.iloc[self.current_step]['Close']
        transaction_cost = 0

        if action == 1:  # Buy
            shares_bought = self.balance // current_price
            transaction_cost = shares_bought * current_price * self.transaction_cost
            self.balance -= shares_bought * current_price + transaction_cost
            self.shares_held += shares_bought

        elif action == 2:  # Sell
            transaction_cost = self.shares_held * current_price * self.transaction_cost
            self.balance += self.shares_held * current_price - transaction_cost
            self.shares_held = 0

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def step(self, action):
        """
        Take a step in the environment using the given action and return the new state, reward, done flag, and info.
        
        Parameters:
        -----------
        action : int
            The action to take (0: Hold, 1: Buy, 2: Sell).
        
        Returns:
        --------
        tuple
            A tuple containing the new observation, reward, done flag, and additional info.
        """
        prev_net_worth = self.net_worth
        self._take_action(action)
        
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            self.current_step = len(self.df) - 1

        # Reward is the change in net worth
        net_worth_change = self.net_worth - prev_net_worth
        reward = net_worth_change

        # Penalty for transaction costs
        reward -= self.transaction_cost * net_worth_change if net_worth_change > 0 else 0

        # Calculate daily return and append to returns list
        daily_return = (self.net_worth - prev_net_worth) / prev_net_worth
        self.returns.append(daily_return)

        # Calculate Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        if len(self.returns) > 1:
            sharpe_ratio = np.mean(self.returns) / (np.std(self.returns) + 1e-9)  # Adding epsilon to avoid division by zero
            reward += sharpe_ratio

        # Done flag to indicate the end of the episode
        done = self.net_worth <= 0 or self.current_step >= len(self.df) - 1

        # Get the next observation
        obs = self._next_observation()
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        """
        Render the current state of the environment.
        
        Parameters:
        -----------
        mode : str
            The mode to render with (currently only 'human' mode is supported).
        close : bool
            Whether to close the rendering (not used in this implementation).
        """
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total worth: {self.shares_held * self.df.iloc[self.current_step]["Close"]})')
        print(f'Net worth: {self.net_worth}')
        print(f'Max net worth: {self.max_net_worth}')
        print(f'Profit: {profit}')
        print("*************************************************************\n")

    def output(self):
        """
        Output final metrics after the training has been completed.

        Returns:
        --------
        dict
            A dictionary containing the final net worth, profit per trade, Sharpe ratio, volatility, and risk of ruin.
        """
        final_net_worth = self.net_worth
        trades = max(len(self.returns) - 1, 1)
        profit_per_trade = (self.net_worth - self.initial_balance) / trades

        # Sharpe ratio calculation (risk-free rate = 0)
        sharpe_ratio = np.mean(self.returns) / (np.std(self.returns) + 1e-9) if len(self.returns) > 1 else 0

        # Volatility (standard deviation of returns)
        volatility = np.std(self.returns) if len(self.returns) > 1 else 0

        # Risk of ruin: Proportion of returns that are negative
        risk_of_ruin = len([r for r in self.returns if r < 0]) / trades if trades > 0 else 0

        return {
            "final_net_worth": final_net_worth,
            "profit_per_trade": profit_per_trade,
            "sharpe_ratio": sharpe_ratio,
            "volatility": volatility,
            "risk_of_ruin": risk_of_ruin
        }





class TradingEnvCCI10(gym.Env):
    """
    A custom trading environment for reinforcement learning in OpenAI's Gym framework.
    
    This environment simulates trading using a given historical price data (in DataFrame format)
    and allows the agent to take actions to buy, sell, or hold the asset.
    
    Attributes:
    -----------
    df : DataFrame
        The historical price data containing at least 'Close' and 'CCI' columns.
    current_step : int
        The current time step in the simulation.
    balance : float
        The current balance in the agent's account.
    shares_held : int
        The number of shares currently held by the agent.
    net_worth : float
        The net worth of the agent's account (balance + value of shares held).
    max_net_worth : float
        The maximum net worth achieved during the simulation.
    initial_balance : float
        The initial balance at the start of the simulation.
    transaction_cost : float
        The cost of making a transaction (e.g., 0.1% of the trade value).
    initial_price : float
        The initial closing price at the start of the simulation.
    returns : list
        A list to track daily returns for calculating the Sharpe ratio.
    """
    
    def __init__(self, df):
        """
        Initialize the trading environment with the given DataFrame of historical price data.

        Parameters:
        -----------
        df : DataFrame
            The historical price data containing 'Close' and 'CCI' columns.
        """
        self.df = df
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_net_worth = self.balance
        self.initial_balance = self.balance

        # Normalize the CCI values
        self.df['CCI'] = (self.df['CCI'] - self.df['CCI'].mean()) / self.df['CCI'].std()

        # Action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: CCI values of the last 10 days
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

        # Example transaction cost: 0.1% per trade
        self.transaction_cost = 0.001
        self.initial_price = self.df.iloc[0]['Close']
        self.returns = []

    def reset(self):
        """
        Reset the environment to the initial state and return the initial observation.
        
        Returns:
        --------
        np.ndarray
            The initial observation of the environment (CCI values of the last 10 days).
        """
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_net_worth = self.balance
        self.returns = []
        return self._next_observation()
    
    def _next_observation(self):
        """
        Get the next observation, which consists of the CCI values of the last 10 days.
        
        Returns:
        --------
        np.ndarray
            The CCI values of the last 10 days.
        """
        start_idx = max(0, self.current_step - 9)
        end_idx = self.current_step + 1
        obs = self.df['CCI'].iloc[start_idx:end_idx].values
        
        # If fewer than 10 days, pad with the first available CCI value
        if len(obs) < 10:
            padding = np.full((10 - len(obs),), obs[0])
            obs = np.concatenate((padding, obs))
        
        return obs
    
    def _take_action(self, action):
        """
        Execute the given action (buy, sell, hold) and update the environment state accordingly.
        
        Parameters:
        -----------
        action : int
            The action to take (0: Hold, 1: Buy, 2: Sell).
        """
        current_price = self.df.iloc[self.current_step]['Close']
        transaction_cost = 0

        if action == 1:  # Buy
            shares_bought = self.balance // current_price
            transaction_cost = shares_bought * current_price * self.transaction_cost
            self.balance -= shares_bought * current_price + transaction_cost
            self.shares_held += shares_bought

        elif action == 2:  # Sell
            transaction_cost = self.shares_held * current_price * self.transaction_cost
            self.balance += self.shares_held * current_price - transaction_cost
            self.shares_held = 0

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def step(self, action):
        """
        Take a step in the environment using the given action and return the new state, reward, done flag, and info.
        
        Parameters:
        -----------
        action : int
            The action to take (0: Hold, 1: Buy, 2: Sell).
        
        Returns:
        --------
        tuple
            A tuple containing the new observation, reward, done flag, and additional info.
        """
        prev_net_worth = self.net_worth
        self._take_action(action)
        
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            self.current_step = len(self.df) - 1

        # Reward is the change in net worth
        net_worth_change = self.net_worth - prev_net_worth
        reward = net_worth_change

        # Penalty for transaction costs
        reward -= self.transaction_cost * net_worth_change if net_worth_change > 0 else 0

        # Calculate daily return and append to returns list
        daily_return = (self.net_worth - prev_net_worth) / prev_net_worth
        self.returns.append(daily_return)

        # Calculate Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        if len(self.returns) > 1:
            sharpe_ratio = np.mean(self.returns) / (np.std(self.returns) + 1e-9)  # Adding epsilon to avoid division by zero
            reward += sharpe_ratio

        # Done flag to indicate the end of the episode
        done = self.net_worth <= 0 or self.current_step >= len(self.df) - 1

        # Get the next observation
        obs = self._next_observation()
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        """
        Render the current state of the environment.
        
        Parameters:
        -----------
        mode : str
            The mode to render with (currently only 'human' mode is supported).
        close : bool
            Whether to close the rendering (not used in this implementation).
        """
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total worth: {self.shares_held * self.df.iloc[self.current_step]["Close"]})')
        print(f'Net worth: {self.net_worth}')
        print(f'Max net worth: {self.max_net_worth}')
        print(f'Profit: {profit}')
        print("*************************************************************\n")

    def output(self):
        """
        Output the final statistics after the training has been completed.

        Returns:
        --------
        dict
            A dictionary containing final_net_worth, profit_per_trade, sharpe_ratio, volatility, and risk_of_ruin.
        """
        final_net_worth = self.net_worth
        total_trades = len(self.returns)
        profit_per_trade = (self.net_worth - self.initial_balance) / total_trades if total_trades > 0 else 0
        sharpe_ratio = np.mean(self.returns) / (np.std(self.returns) + 1e-9) if len(self.returns) > 1 else 0
        volatility = np.std(self.returns)
        risk_of_ruin = sum(1 for r in self.returns if r < -0.1) / total_trades if total_trades > 0 else 0  # Example threshold: 10% loss

        return {
            "final_net_worth": final_net_worth,
            "profit_per_trade": profit_per_trade,
            "sharpe_ratio": sharpe_ratio,
            "volatility": volatility,
            "risk_of_ruin": risk_of_ruin
        }



class TradingEnvFiveFive(gym.Env):
    """
    A custom trading environment for reinforcement learning in OpenAI's Gym framework.
    
    This environment simulates trading using a given historical price data (in DataFrame format)
    and allows the agent to take actions to buy, sell, or hold the asset.
    
    Attributes:
    -----------
    df : DataFrame
        The historical price data containing 'CCI', 'WCLPrice', 'SAREXT', 'Slowd', 'TypePrice', and 'Close' columns.
    current_step : int
        The current time step in the simulation.
    balance : float
        The current balance in the agent's account.
    shares_held : int
        The number of shares currently held by the agent.
    net_worth : float
        The net worth of the agent's account (balance + value of shares held).
    max_net_worth : float
        The maximum net worth achieved during the simulation.
    initial_balance : float
        The initial balance at the start of the simulation.
    transaction_cost : float
        The cost of making a transaction (e.g., 0.1% of the trade value).
    initial_price : float
        The initial closing price at the start of the simulation.
    returns : list
        A list to track daily returns for calculating the Sharpe ratio.
    scaler : StandardScaler
        Scaler for normalizing the observation data.
    """
    
    def __init__(self, df):
        """
        Initialize the trading environment with the given DataFrame of historical price data.

        Parameters:
        -----------
        df : DataFrame
            The historical price data containing 'CCI', 'WCLPrice', 'SAREXT', 'Slowd', 'TypePrice', and 'Close' columns.
        """
        self.df = df
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_net_worth = self.balance
        self.initial_balance = self.balance

        # Action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: values of the specified columns for the last 5 days (excluding 'Close')
        self.observation_space = spaces.Box(low=0, high=1, shape=(25,), dtype=np.float32)

        # Example transaction cost: 0.1% per trade
        self.transaction_cost = 0.001
        self.initial_price = self.df.iloc[0]['Close']
        self.returns = []

        # Normalizing the observation data (excluding 'Close')
        self.scaler = StandardScaler()
        self.df[['CCI', 'WCLPrice', 'SAREXT', 'Slowd', 'TypePrice']] = self.scaler.fit_transform(self.df[['CCI', 'WCLPrice', 'SAREXT', 'Slowd', 'TypePrice']])

    def reset(self):
        """
        Reset the environment to the initial state and return the initial observation.
        
        Returns:
        --------
        np.ndarray
            The initial observation of the environment (values of the specified columns).
        """
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_net_worth = self.balance
        self.returns = []
        return self._next_observation()
    
    def _next_observation(self):
        """
        Get the next observation, which consists of the normalized values of the specified columns for the last 5 days.
        
        Returns:
        --------
        np.ndarray
            The normalized values of the specified columns for the last 5 days.
        """
        start_idx = max(0, self.current_step - 4)
        end_idx = self.current_step + 1
        obs = self.df[['CCI', 'WCLPrice', 'SAREXT', 'Slowd', 'TypePrice']].iloc[start_idx:end_idx].values.flatten()

        # If fewer than 5 days, pad with the first available values
        if len(obs) < 25:
            padding = np.tile(obs[:5], 5 - (end_idx - start_idx))
            obs = np.concatenate((padding, obs))
        
        return obs
    
    def _take_action(self, action):
        """
        Execute the given action (buy, sell, hold) and update the environment state accordingly.
        
        Parameters:
        -----------
        action : int
            The action to take (0: Hold, 1: Buy, 2: Sell).
        """
        current_price = self.df.iloc[self.current_step]['Close']
        transaction_cost = 0

        if action == 1:  # Buy
            shares_bought = self.balance // current_price
            transaction_cost = shares_bought * current_price * self.transaction_cost
            self.balance -= shares_bought * current_price + transaction_cost
            self.shares_held += shares_bought

        elif action == 2:  # Sell
            transaction_cost = self.shares_held * current_price * self.transaction_cost
            self.balance += self.shares_held * current_price - transaction_cost
            self.shares_held = 0

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def step(self, action):
        """
        Take a step in the environment using the given action and return the new state, reward, done flag, and info.
        
        Parameters:
        -----------
        action : int
            The action to take (0: Hold, 1: Buy, 2: Sell).
        
        Returns:
        --------
        tuple
            A tuple containing the new observation, reward, done flag, and additional info.
        """
        prev_net_worth = self.net_worth
        self._take_action(action)
        
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            self.current_step = len(self.df) - 1

        # Reward is the change in net worth
        net_worth_change = self.net_worth - prev_net_worth
        reward = net_worth_change

        # Penalty for transaction costs
        reward -= self.transaction_cost * net_worth_change if net_worth_change > 0 else 0

        # Calculate daily return and append to returns list
        daily_return = (self.net_worth - prev_net_worth) / prev_net_worth
        self.returns.append(daily_return)

        # Calculate Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        if len(self.returns) > 1:
            sharpe_ratio = np.mean(self.returns) / (np.std(self.returns) + 1e-9)  # Adding epsilon to avoid division by zero
            reward += sharpe_ratio

        # Done flag to indicate the end of the episode
        done = self.net_worth <= 0 or self.current_step >= len(self.df) - 1

        # Get the next observation
        obs = self._next_observation()
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        """
        Render the current state of the environment.
        
        Parameters:
        -----------
        mode : str
            The mode to render with (currently only 'human' mode is supported).
        close : bool
            Whether to close the rendering (not used in this implementation).
        """
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total worth: {self.shares_held * self.df.iloc[self.current_step]["Close"]})')
        print(f'Net worth: {self.net_worth}')
        print(f'Max net worth: {self.max_net_worth}')
        print(f'Profit: {profit}')
        print("*************************************************************\n")

    def output(self):
        """
        Output the final statistics after the training has been completed.

        Returns:
        --------
        dict
            A dictionary containing final_net_worth, profit_per_trade, sharpe_ratio, volatility, and risk_of_ruin.
        """
        final_net_worth = self.net_worth
        total_trades = len(self.returns)
        profit_per_trade = (self.net_worth - self.initial_balance) / total_trades if total_trades > 0 else 0
        sharpe_ratio = np.mean(self.returns) / (np.std(self.returns) + 1e-9) if len(self.returns) > 1 else 0
        volatility = np.std(self.returns)
        risk_of_ruin = sum(1 for r in self.returns if r < -0.1) / total_trades if total_trades > 0 else 0  # Example threshold: 10% loss

        return {
            "final_net_worth": final_net_worth,
            "profit_per_trade": profit_per_trade,
            "sharpe_ratio": sharpe_ratio,
            "volatility": volatility,
            "risk_of_ruin": risk_of_ruin
        }


class TradingEnvAlterFiveFive(gym.Env):
    """
    A custom trading environment for reinforcement learning in OpenAI's Gym framework.
    
    This environment simulates trading using a given historical price data (in DataFrame format)
    and allows the agent to take actions to buy, sell, or hold the asset.
    
    Attributes:
    -----------
    df : DataFrame
        The historical price data containing 'ADX', 'RSI', 'EMA', 'MACD', 'OBV', and 'Close' columns.
    current_step : int
        The current time step in the simulation.
    balance : float
        The current balance in the agent's account.
    shares_held : int
        The number of shares currently held by the agent.
    net_worth : float
        The net worth of the agent's account (balance + value of shares held).
    max_net_worth : float
        The maximum net worth achieved during the simulation.
    initial_balance : float
        The initial balance at the start of the simulation.
    transaction_cost : float
        The cost of making a transaction (e.g., 0.1% of the trade value).
    initial_price : float
        The initial closing price at the start of the simulation.
    returns : list
        A list to track daily returns for calculating the Sharpe ratio.
    scaler : StandardScaler
        Scaler for normalizing the observation data.
    """
    
    def __init__(self, df):
        """
        Initialize the trading environment with the given DataFrame of historical price data.

        Parameters:
        -----------
        df : DataFrame
            The historical price data containing 'ADX', 'RSI', 'EMA', 'MACD', 'OBV', and 'Close' columns.
        """
        self.df = df
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_net_worth = self.balance
        self.initial_balance = self.balance

        # Action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: values of the specified columns for the last 5 days (excluding 'Close')
        self.observation_space = spaces.Box(low=0, high=1, shape=(25,), dtype=np.float32)

        # Example transaction cost: 0.1% per trade
        self.transaction_cost = 0.001
        self.initial_price = self.df.iloc[0]['Close']
        self.returns = []

        # Normalizing the observation data (excluding 'Close')
        self.scaler = StandardScaler()
        self.df[['ADX', 'RSI', 'EMA', 'MACD', 'OBV']] = self.scaler.fit_transform(self.df[['ADX', 'RSI', 'EMA', 'MACD', 'OBV']])

    def reset(self):
        """
        Reset the environment to the initial state and return the initial observation.
        
        Returns:
        --------
        np.ndarray
            The initial observation of the environment (values of the specified columns).
        """
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_net_worth = self.balance
        self.returns = []
        return self._next_observation()
    
    def _next_observation(self):
        """
        Get the next observation, which consists of the normalized values of the specified columns for the last 5 days.
        
        Returns:
        --------
        np.ndarray
            The normalized values of the specified columns for the last 5 days.
        """
        start_idx = max(0, self.current_step - 4)
        end_idx = self.current_step + 1
        obs = self.df[['ADX', 'RSI', 'EMA', 'MACD', 'OBV']].iloc[start_idx:end_idx].values.flatten()

        # If fewer than 5 days, pad with the first available values
        if len(obs) < 25:
            padding = np.tile(obs[:5], 5 - (end_idx - start_idx))
            obs = np.concatenate((padding, obs))
        
        return obs
    
    def _take_action(self, action):
        """
        Execute the given action (buy, sell, hold) and update the environment state accordingly.
        
        Parameters:
        -----------
        action : int
            The action to take (0: Hold, 1: Buy, 2: Sell).
        """
        current_price = self.df.iloc[self.current_step]['Close']
        transaction_cost = 0

        if action == 1:  # Buy
            shares_bought = self.balance // current_price
            transaction_cost = shares_bought * current_price * self.transaction_cost
            self.balance -= shares_bought * current_price + transaction_cost
            self.shares_held += shares_bought

        elif action == 2:  # Sell
            transaction_cost = self.shares_held * current_price * self.transaction_cost
            self.balance += self.shares_held * current_price - transaction_cost
            self.shares_held = 0

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def step(self, action):
        """
        Take a step in the environment using the given action and return the new state, reward, done flag, and info.
        
        Parameters:
        -----------
        action : int
            The action to take (0: Hold, 1: Buy, 2: Sell).
        
        Returns:
        --------
        tuple
            A tuple containing the new observation, reward, done flag, and additional info.
        """
        prev_net_worth = self.net_worth
        self._take_action(action)
        
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            self.current_step = len(self.df) - 1

        # Reward is the change in net worth
        net_worth_change = self.net_worth - prev_net_worth
        reward = net_worth_change

        # Penalty for transaction costs
        reward -= self.transaction_cost * net_worth_change if net_worth_change > 0 else 0

        # Calculate daily return and append to returns list
        daily_return = (self.net_worth - prev_net_worth) / prev_net_worth
        self.returns.append(daily_return)

        # Calculate Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        if len(self.returns) > 1:
            sharpe_ratio = np.mean(self.returns) / (np.std(self.returns) + 1e-9)  # Adding epsilon to avoid division by zero
            reward += sharpe_ratio

        # Done flag to indicate the end of the episode
        done = self.net_worth <= 0 or self.current_step >= len(self.df) - 1

        # Get the next observation
        obs = self._next_observation()
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        """
        Render the current state of the environment.
        
        Parameters:
        -----------
        mode : str
            The mode to render with (currently only 'human' mode is supported).
        close : bool
            Whether to close the rendering (not used in this implementation).
        """
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total worth: {self.shares_held * self.df.iloc[self.current_step]["Close"]})')
        print(f'Net worth: {self.net_worth}')
        print(f'Max net worth: {self.max_net_worth}')
        print(f'Profit: {profit}')
        print("*************************************************************\n")

    def output(self):
        """
        Output the final statistics after the training has been completed.

        Returns:
        --------
        dict
            A dictionary containing final_net_worth, profit_per_trade, sharpe_ratio, volatility, and risk_of_ruin.
        """
        final_net_worth = self.net_worth
        total_trades = len(self.returns)
        profit_per_trade = (self.net_worth - self.initial_balance) / total_trades if total_trades > 0 else 0
        sharpe_ratio = np.mean(self.returns) / (np.std(self.returns) + 1e-9) if len(self.returns) > 1 else 0
        volatility = np.std(self.returns)
        risk_of_ruin = sum(1 for r in self.returns if r < -0.1) / total_trades if total_trades > 0 else 0  # Example threshold: 10% loss

        return {
            "final_net_worth": final_net_worth,
            "profit_per_trade": profit_per_trade,
            "sharpe_ratio": sharpe_ratio,
            "volatility": volatility,
            "risk_of_ruin": risk_of_ruin
        }


class TradingEnv(gym.Env):
    """
    A custom trading environment for reinforcement learning in OpenAI's Gym framework.
    
    This environment simulates trading using historical price data and allows the agent to take actions
    to buy, sell, or hold the asset.
    
    Attributes:
    -----------
    df : DataFrame
        The historical price data containing 'Close', 'Open', 'High', 'Low', and 'Volume' columns.
    current_step : int
        The current time step in the simulation.
    balance : float
        The current balance in the agent's account.
    shares_held : int
        The number of shares currently held by the agent.
    net_worth : float
        The net worth of the agent's account (balance + value of shares held).
    max_net_worth : float
        The maximum net worth achieved during the simulation.
    initial_balance : float
        The initial balance at the start of the simulation.
    transaction_cost : float
        The cost of making a transaction (e.g., 0.1% of the trade value).
    returns : list
        A list to track daily returns for calculating the Sharpe ratio.
    scaler : StandardScaler
        Scaler for normalizing the observation data.
    """
    
    def __init__(self, df):
        """
        Initialize the trading environment with the given DataFrame of historical price data.

        Parameters:
        -----------
        df : DataFrame
            The historical price data containing 'Close', 'Open', 'High', 'Low', and 'Volume' columns.
        """
        super(TradingEnv, self).__init__()
        
        self.df = df.copy()
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_net_worth = self.balance
        self.initial_balance = self.balance

        # Action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: normalized values of 'Open', 'High', 'Low', 'Volume' and raw 'Close'
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,), dtype=np.float32)

        # Example transaction cost: 0.1% per trade
        self.transaction_cost = 0.001
        self.initial_price = self.df.iloc[0]['Close']
        self.returns = []

        # Normalize the observation data
        self.scaler = StandardScaler()
        self.df[['Open', 'High', 'Low', 'Volume']] = self.scaler.fit_transform(self.df[['Open', 'High', 'Low', 'Volume']])

    def reset(self):
        """
        Reset the environment to the initial state and return the initial observation.
        
        Returns:
        --------
        np.ndarray
            The initial observation of the environment (normalized values of 'Open', 'High', 'Low', 'Volume' and raw 'Close').
        """
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_net_worth = self.balance
        self.returns = []
        return self._next_observation()
    
    def _next_observation(self):
        """
        Get the next observation, which consists of normalized values of 'Open', 'High', 'Low', 'Volume' and raw 'Close'.
        
        Returns:
        --------
        np.ndarray
            The normalized values of 'Open', 'High', 'Low', 'Volume' and raw 'Close'.
        """
        obs = self.df.iloc[self.current_step][['Open', 'High', 'Low', 'Volume']].values
        raw_close = self.df.iloc[self.current_step]['Close']
        return np.append(obs, raw_close)
    
    def _take_action(self, action):
        """
        Execute the given action (buy, sell, hold) and update the environment state accordingly.
        
        Parameters:
        -----------
        action : int
            The action to take (0: Hold, 1: Buy, 2: Sell).
        """
        current_price = self.df.iloc[self.current_step]['Close']
        transaction_cost = 0

        if action == 1:  # Buy
            shares_bought = self.balance // current_price
            transaction_cost = shares_bought * current_price * self.transaction_cost
            self.balance -= shares_bought * current_price + transaction_cost
            self.shares_held += shares_bought

        elif action == 2:  # Sell
            transaction_cost = self.shares_held * current_price * self.transaction_cost
            self.balance += self.shares_held * current_price - transaction_cost
            self.shares_held = 0

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def step(self, action):
        """
        Take a step in the environment using the given action and return the new state, reward, done flag, and info.
        
        Parameters:
        -----------
        action : int
            The action to take (0: Hold, 1: Buy, 2: Sell).
        
        Returns:
        --------
        tuple
            A tuple containing the new observation, reward, done flag, and additional info.
        """
        prev_net_worth = self.net_worth
        self._take_action(action)
        
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            self.current_step = len(self.df) - 1

        # Reward is the change in net worth
        net_worth_change = self.net_worth - prev_net_worth
        reward = net_worth_change

        # Penalty for transaction costs
        reward -= self.transaction_cost * net_worth_change if net_worth_change > 0 else 0

        # Calculate daily return and append to returns list
        daily_return = (self.net_worth - prev_net_worth) / prev_net_worth
        self.returns.append(daily_return)

        # Calculate Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        if len(self.returns) > 1:
            sharpe_ratio = np.mean(self.returns) / (np.std(self.returns) + 1e-9)  # Adding epsilon to avoid division by zero
            reward += sharpe_ratio

        # Done flag to indicate the end of the episode
        done = self.net_worth <= 0 or self.current_step >= len(self.df) - 1

        # Get the next observation
        obs = self._next_observation()
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        """
        Render the current state of the environment.
        
        Parameters:
        -----------
        mode : str
            The mode to render with (currently only 'human' mode is supported).
        close : bool
            Whether to close the rendering (not used in this implementation).
        """
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total worth: {self.shares_held * self.df.iloc[self.current_step]["Close"]})')
        print(f'Net worth: {self.net_worth}')
        print(f'Max net worth: {self.max_net_worth}')
        print(f'Profit: {profit}')
        print("*************************************************************\n")

    def output(self):
        """
        Output the final statistics after the training has been completed.

        Returns:
        --------
        dict
            A dictionary containing final_net_worth, profit_per_trade, sharpe_ratio, volatility, and risk_of_ruin.
        """
        final_net_worth = self.net_worth
        total_trades = len(self.returns)
        profit_per_trade = (self.net_worth - self.initial_balance) / total_trades if total_trades > 0 else 0
        sharpe_ratio = np.mean(self.returns) / (np.std(self.returns) + 1e-9) if len(self.returns) > 1 else 0
        volatility = np.std(self.returns)
        risk_of_ruin = sum(1 for r in self.returns if r < -0.1) / total_trades if total_trades > 0 else 0  # Example threshold: 10% loss

        return {
            "final_net_worth": final_net_worth,
            "profit_per_trade": profit_per_trade,
            "sharpe_ratio": sharpe_ratio,
            "volatility": volatility,
            "risk_of_ruin": risk_of_ruin
        }









class TradingEnvGeneral(gym.Env):
    """
    A custom trading environment for reinforcement learning in OpenAI's Gym framework.
    
    This environment simulates trading using a given historical price data (in DataFrame format)
    and allows the agent to take actions to buy, sell, or hold the asset.
    
    Attributes:
    -----------
    df : DataFrame
        The historical price data containing at least 'Close' and the specified technical indicator columns.
    current_step : int
        The current time step in the simulation.
    balance : float
        The current balance in the agent's account.
    shares_held : int
        The number of shares currently held by the agent.
    net_worth : float
        The net worth of the agent's account (balance + value of shares held).
    max_net_worth : float
        The maximum net worth achieved during the simulation.
    initial_balance : float
        The initial balance at the start of the simulation.
    transaction_cost : float
        The cost of making a transaction (e.g., 0.1% of the trade value).
    initial_price : float
        The initial closing price at the start of the simulation.
    returns : list
        A list to track daily returns for calculating the Sharpe ratio.
    indicator : str
        The technical indicator used for trading (e.g., 'CCI').
    time_period : int
        The number of days to consider for the observation window.
    """
    
    def __init__(self, df, indicator='CCI', time_period=10):
        """
        Initialize the trading environment with the given DataFrame of historical price data.

        Parameters:
        -----------
        df : DataFrame
            The historical price data containing 'Close' and the specified technical indicator columns.
        indicator : str
            The technical indicator used for trading (default: 'CCI').
        time_period : int
            The number of days to consider for the observation window (default: 10).
        """
        self.df = df
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_net_worth = self.balance
        self.initial_balance = self.balance
        self.indicator = indicator
        self.time_period = time_period

        # Normalize the indicator values
        self.df[self.indicator] = (self.df[self.indicator] - self.df[self.indicator].mean()) / self.df[self.indicator].std()

        # Action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: Indicator values of the last 'time_period' days
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.time_period,), dtype=np.float32)

        # Example transaction cost: 0.1% per trade
        self.transaction_cost = 0.001
        self.initial_price = self.df.iloc[0]['Close']
        self.returns = []

    def reset(self):
        """
        Reset the environment to the initial state and return the initial observation.
        
        Returns:
        --------
        np.ndarray
            The initial observation of the environment (indicator values of the last 'time_period' days).
        """
        self.current_step = 0
        self.balance = 10000
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_net_worth = self.balance
        self.returns = []
        return self._next_observation()
    
    def _next_observation(self):
        """
        Get the next observation, which consists of the indicator values of the last 'time_period' days.
        
        Returns:
        --------
        np.ndarray
            The indicator values of the last 'time_period' days.
        """
        start_idx = max(0, self.current_step - self.time_period + 1)
        end_idx = self.current_step + 1
        obs = self.df[self.indicator].iloc[start_idx:end_idx].values
        
        # If fewer than 'time_period' days, pad with the first available indicator value
        if len(obs) < self.time_period:
            padding = np.full((self.time_period - len(obs),), obs[0])
            obs = np.concatenate((padding, obs))
        
        return obs
    
    def _take_action(self, action):
        """
        Execute the given action (buy, sell, hold) and update the environment state accordingly.
        
        Parameters:
        -----------
        action : int
            The action to take (0: Hold, 1: Buy, 2: Sell).
        """
        current_price = self.df.iloc[self.current_step]['Close']
        transaction_cost = 0

        if action == 1:  # Buy
            shares_bought = self.balance // current_price
            transaction_cost = shares_bought * current_price * self.transaction_cost
            self.balance -= shares_bought * current_price + transaction_cost
            self.shares_held += shares_bought

        elif action == 2:  # Sell
            transaction_cost = self.shares_held * current_price * self.transaction_cost
            self.balance += self.shares_held * current_price - transaction_cost
            self.shares_held = 0

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def step(self, action):
        """
        Take a step in the environment using the given action and return the new state, reward, done flag, and info.
        
        Parameters:
        -----------
        action : int
            The action to take (0: Hold, 1: Buy, 2: Sell).
        
        Returns:
        --------
        tuple
            A tuple containing the new observation, reward, done flag, and additional info.
        """
        prev_net_worth = self.net_worth
        self._take_action(action)
        
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            self.current_step = len(self.df) - 1

        # Reward is the change in net worth
        net_worth_change = self.net_worth - prev_net_worth
        reward = net_worth_change

        # Penalty for transaction costs
        reward -= self.transaction_cost * net_worth_change if net_worth_change > 0 else 0

        # Calculate daily return and append to returns list
        daily_return = (self.net_worth - prev_net_worth) / prev_net_worth
        self.returns.append(daily_return)

        # Calculate Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        if len(self.returns) > 1:
            sharpe_ratio = np.mean(self.returns) / (np.std(self.returns) + 1e-9)  # Adding epsilon to avoid division by zero
            reward += sharpe_ratio

        # Done flag to indicate the end of the episode
        done = self.net_worth <= 0 or self.current_step >= len(self.df) - 1

        # Get the next observation
        obs = self._next_observation()
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        """
        Render the current state of the environment.
        
        Parameters:
        -----------
        mode : str
            The mode to render with (currently only 'human' mode is supported).
        close : bool
            Whether to close the rendering (not used in this implementation).
        """
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total worth: {self.shares_held * self.df.iloc[self.current_step]["Close"]})')
        print(f'Net worth: {self.net_worth}')
        print(f'Max net worth: {self.max_net_worth}')
        print(f'Profit: {profit}')
        print("*************************************************************\n")

    def output(self):
        """
        Output the final statistics after the training has been completed.

        Returns:
        --------
        dict
            A dictionary containing final_net_worth, profit_per_trade, sharpe_ratio, volatility, and risk_of_ruin.
        """
        final_net_worth = self.net_worth
        total_trades = len(self.returns)
        profit_per_trade = (self.net_worth - self.initial_balance) / total_trades if total_trades > 0 else 0
        sharpe_ratio = np.mean(self.returns) / (np.std(self.returns) + 1e-9) if len(self.returns) > 1 else 0
        volatility = np.std(self.returns)
        risk_of_ruin = sum(1 for r in self.returns if r < -0.1) / total_trades if total_trades > 0 else 0  # Example threshold: 10% loss

        return {
            "final_net_worth": final_net_worth,
            "profit_per_trade": profit_per_trade,
            "sharpe_ratio": sharpe_ratio,
            "volatility": volatility,
            "risk_of_ruin": risk_of_ruin
        }




class TradingEnvGeneral2(gym.Env):
    """
    A custom trading environment for reinforcement learning in OpenAI's Gym framework.
    
    This environment simulates trading using a given historical price data (in DataFrame format)
    and allows the agent to take actions to buy, sell, or hold the asset.
    
    Attributes:
    -----------
    df : DataFrame
        The historical price data containing custom technical indicators and 'Close' column.
    indicators : list
        The list of technical indicators used in the observation space.
    time_period : int
        The number of days to consider for the observation space.
    current_step : int
        The current time step in the simulation.
    balance : float
        The current balance in the agent's account.
    shares_held : int
        The number of shares currently held by the agent.
    net_worth : float
        The net worth of the agent's account (balance + value of shares held).
    max_net_worth : float
        The maximum net worth achieved during the simulation.
    initial_balance : float
        The initial balance at the start of the simulation.
    transaction_cost : float
        The cost of making a transaction (e.g., 0.1% of the trade value).
    initial_price : float
        The initial closing price at the start of the simulation.
    returns : list
        A list to track daily returns for calculating the Sharpe ratio.
    scaler : StandardScaler
        Scaler for normalizing the observation data.
    transactions : list
        A list to keep track of transactions (buy/sell actions).
    """
    
    def __init__(self, df, indicators, time_period=5, initial_balance=10000):
        """
        Initialize the trading environment with the given DataFrame of historical price data.

        Parameters:
        -----------
        df : DataFrame
            The historical price data containing custom technical indicators and 'Close' column.
        indicators : list
            The list of technical indicators used in the observation space.
        time_period : int
            The number of days to consider for the observation space.
        initial_balance : float
            The initial balance at the start of the simulation.
        """
        self.df = df
        self.indicators = indicators
        self.time_period = time_period
        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_net_worth = self.balance
        self.initial_balance = self.balance

        # Action space: 0: Hold, 1: Buy, 2: Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: values of the specified indicators for the last `time_period` days (excluding 'Close')
        self.observation_space = spaces.Box(
            low=0, 
            high=1, 
            shape=(len(indicators) * time_period,), 
            dtype=np.float32
        )

        # Example transaction cost: 0.1% per trade
        self.transaction_cost = 0.001
        self.initial_price = self.df.iloc[0]['Close']
        self.returns = []

        # Normalizing the observation data (excluding 'Close')
        self.scaler = StandardScaler()
        self.df[indicators] = self.scaler.fit_transform(self.df[indicators])

        self.transactions = []

    def reset(self):
        """
        Reset the environment to the initial state and return the initial observation.
        
        Returns:
        --------
        np.ndarray
            The initial observation of the environment (values of the specified columns).
        """
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.balance
        self.max_net_worth = self.balance
        self.returns = []
        self.transactions = []
        return self._next_observation()
    
    def _next_observation(self):
        """
        Get the next observation, which consists of the normalized values of the specified indicators for the last `time_period` days.
        
        Returns:
        --------
        np.ndarray
            The normalized values of the specified indicators for the last `time_period` days.
        """
        start_idx = max(0, self.current_step - self.time_period + 1)
        end_idx = self.current_step + 1
        obs = self.df[self.indicators].iloc[start_idx:end_idx].values.flatten()

        # If fewer than `time_period` days, pad with the first available values
        if len(obs) < len(self.indicators) * self.time_period:
            padding = np.tile(obs[:len(self.indicators)], self.time_period - (end_idx - start_idx))
            obs = np.concatenate((padding, obs))
        
        return obs
    
    def _take_action(self, action):
        """
        Execute the given action (buy, sell, hold) and update the environment state accordingly.
        
        Parameters:
        -----------
        action : int
            The action to take (0: Hold, 1: Buy, 2: Sell).
        """
        current_price = self.df.iloc[self.current_step]['Close']
        transaction_cost = 0

        if action == 1:  # Buy
            shares_bought = self.balance // current_price
            if shares_bought > 0:
                transaction_cost = shares_bought * current_price * self.transaction_cost
                self.balance -= shares_bought * current_price + transaction_cost
                self.shares_held += shares_bought
                self.transactions.append(('buy', shares_bought, current_price))

        elif action == 2:  # Sell
            if self.shares_held > 0:
                transaction_cost = self.shares_held * current_price * self.transaction_cost
                self.balance += self.shares_held * current_price - transaction_cost
                self.transactions.append(('sell', self.shares_held, current_price))
                self.shares_held = 0

        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth

    def step(self, action):
        """
        Take a step in the environment using the given action and return the new state, reward, done flag, and info.
        
        Parameters:
        -----------
        action : int
            The action to take (0: Hold, 1: Buy, 2: Sell).
        
        Returns:
        --------
        tuple
            A tuple containing the new observation, reward, done flag, and additional info.
        """
        prev_net_worth = self.net_worth
        self._take_action(action)
        
        self.current_step += 1

        if self.current_step >= len(self.df) - 1:
            self.current_step = len(self.df) - 1

        # Reward is the change in net worth
        net_worth_change = self.net_worth - prev_net_worth
        reward = net_worth_change

        # Penalty for transaction costs
        reward -= self.transaction_cost * net_worth_change if net_worth_change > 0 else 0

        # Calculate daily return and append to returns list
        daily_return = (self.net_worth - prev_net_worth) / prev_net_worth
        self.returns.append(daily_return)

        # Calculate Sharpe ratio (assuming risk-free rate = 0 for simplicity)
        if len(self.returns) > 1:
            sharpe_ratio = np.mean(self.returns) / (np.std(self.returns) + 1e-9)  # Adding epsilon to avoid division by zero
            reward += sharpe_ratio

        # Done flag to indicate the end of the episode
        done = self.net_worth <= 0 or self.current_step >= len(self.df) - 1

        # Get the next observation
        obs = self._next_observation()
        return obs, reward, done, {}

    def render(self, mode='human', close=False):
        """
        Render the current state of the environment.
        
        Parameters:
        -----------
        mode : str
            The mode to render with (currently only 'human' mode is supported).
        close : bool
            Whether to close the rendering (not used in this implementation).
        """
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held} (Total worth: {self.shares_held * self.df.iloc[self.current_step]["Close"]})')
        print(f'Net worth: {self.net_worth}')
        print(f'Max net worth: {self.max_net_worth}')
        print(f'Profit: {profit}')
        print("*************************************************************\n")

    def output(self):
        """
        Output the final statistics after the training has been completed.

        Returns:
        --------
        dict
            A dictionary containing final_net_worth, profit_per_trade, sharpe_ratio, volatility, and risk_of_ruin.
        """
        final_net_worth = self.net_worth
        total_trades = len(self.returns)
        profit_per_trade = (self.net_worth - self.initial_balance) / total_trades if total_trades > 0 else 0
        sharpe_ratio = np.mean(self.returns) / (np.std(self.returns) + 1e-9) if len(self.returns) > 1 else 0
        volatility = np.std(self.returns)
        risk_of_ruin = sum(1 for r in self.returns if r < -0.1) / total_trades if total_trades > 0 else 0  # Example threshold: 10% loss

        return {
            "final_net_worth": final_net_worth,
            "profit_per_trade": profit_per_trade,
            "sharpe_ratio": sharpe_ratio,
            "volatility": volatility,
            "risk_of_ruin": risk_of_ruin
        }
