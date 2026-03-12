from stable_baselines3 import PPO
import pandas as pd
from portfolio_env import PortfolioEnv
from config import *
from model import CustomCNN

def load_trained_model():
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=FEATURES_DIM),
    )
    model = PPO.load(MODEL_PATH, policy_kwargs=policy_kwargs)
    return model

def get_allocation_for_latest_row(df):
    price_cols = TICKERS
    signal_cols = [c for c in df.columns if c not in price_cols]

    model = load_trained_model()

    env = PortfolioEnv(
        data_df=df,
        ticker_list=price_cols,
        signal_list=signal_cols,
        window_length=WINDOW_LENGTH,
        start_date_index=len(df)-WINDOW_LENGTH-1,
        steps=1
    )

    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)

    weights = action.tolist()
    return weights
