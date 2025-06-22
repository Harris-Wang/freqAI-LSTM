import logging
from functools import reduce
from typing import Dict
import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib

from freqtrade.exchange.exchange_utils import *
from freqtrade.strategy import IStrategy, RealParameter

logger = logging.getLogger(__name__)


class LSTM2Strategy(IStrategy):
    """
    优化版LSTM策略 - 多空分离评价系统
    
    主要改进：
    1. 多空独立目标评价体系
    2. 简化权重参数系统
    3. 增强特征工程
    4. 改进风险控制
    """
    
    # 简化后的超参数空间
    buy_params = {
        "threshold_long": 0.65,
        "threshold_long_exit": 0.25,
        "momentum_weight": 0.4,
        "trend_weight": 0.35,
        "volatility_weight": 0.25,
    }

    sell_params = {
        "threshold_short": -0.6,
        "threshold_short_exit": 0.2,
        "short_momentum_weight": 0.45,
        "short_volatility_weight": 0.3,
        "short_structure_weight": 0.25,
    }

    # ROI和止损设置
    minimal_roi = {
        "0": 0.15,   # 15%目标收益
        "300": 0.08, # 5分钟后8%
        "600": 0.04, # 10分钟后4%
        "1200": 0    # 20分钟后让模型决定
    }

    stoploss = -0.08  # 8%止损

    # 追踪止损
    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.04
    trailing_only_offset_is_reached = True

    timeframe = "1h"
    can_short = True
    use_exit_signal = True
    process_only_new_candles = True
    startup_candle_count = 100  # 增加启动蜡烛数

    # 简化的参数定义
    threshold_long = RealParameter(0.3, 1.0, default=0.65, space='buy')
    threshold_long_exit = RealParameter(0.1, 0.5, default=0.25, space='buy')
    threshold_short = RealParameter(-1.0, -0.3, default=-0.6, space='sell')
    threshold_short_exit = RealParameter(0.1, 0.5, default=0.2, space='sell')
    
    # 三大类权重（总和为1）
    momentum_weight = RealParameter(0.2, 0.6, default=0.4, space='buy')
    trend_weight = RealParameter(0.2, 0.5, default=0.35, space='buy')
    volatility_weight = RealParameter(0.1, 0.4, default=0.25, space='buy')
    
    # 做空专用权重
    short_momentum_weight = RealParameter(0.3, 0.6, default=0.45, space='sell')
    short_volatility_weight = RealParameter(0.2, 0.4, default=0.3, space='sell')
    short_structure_weight = RealParameter(0.15, 0.35, default=0.25, space='sell')

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: Dict, **kwargs):
        """
        扩展特征工程 - 优化指标计算
        """
        # 核心技术指标
        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=20)
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=14)  # 标准14周期
        dataframe["%-momentum-period"] = ta.MOM(dataframe, timeperiod=10)  # 延长周期
        dataframe['%-ma-period'] = ta.SMA(dataframe, timeperiod=20)  # 标准20周期
        
        # MACD系统
        dataframe['%-macd-period'], dataframe['%-macdsignal-period'], dataframe['%-macdhist-period'] = ta.MACD(
            dataframe['close'], slowperiod=26, fastperiod=12, signalperiod=9)
        
        # 变化率
        dataframe['%-roc-period'] = ta.ROC(dataframe, timeperiod=5)  # 延长周期
        
        # 布林带系统
        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=20, stds=2.0
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]
        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"] - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = (
            dataframe["close"] / dataframe["bb_lowerband-period"]
        )
        
        # 新增：威廉指标
        dataframe["%-willr-period"] = ta.WILLR(dataframe, timeperiod=14)
        
        # 新增：随机指标
        stoch = ta.STOCH(dataframe, fastk_period=14, slowk_period=3, slowd_period=3)
        dataframe["%-stoch_k-period"] = stoch['slowk']
        dataframe["%-stoch_d-period"] = stoch['slowd']

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs):
        """
        基础特征工程
        """
        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]
        
        # 新增：价格相对位置
        dataframe["%-price_position"] = (
            dataframe["close"] - dataframe["low"].rolling(window=20).min()
        ) / (
            dataframe["high"].rolling(window=20).max() - dataframe["low"].rolling(window=20).min()
        )
        
        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: Dict, **kwargs):
        """
        标准特征工程 - 时间特征
        """
        dataframe['date'] = pd.to_datetime(dataframe['date'])
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        
        # 新增：月份特征（捕捉季节性）
        dataframe["%-month"] = dataframe["date"].dt.month
        
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:
        """
        优化的目标构建系统 - 多空分离评价
        """
        # ===== 基础指标计算 =====
        dataframe['ma_20'] = ta.SMA(dataframe, timeperiod=20)
        dataframe['ma_50'] = ta.SMA(dataframe, timeperiod=50)
        dataframe['ma_100'] = ta.SMA(dataframe, timeperiod=100)
        
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['willr'] = ta.WILLR(dataframe, timeperiod=14)
        
        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = ta.MACD(
            dataframe['close'], slowperiod=26, fastperiod=12, signalperiod=9)
        
        dataframe['momentum'] = ta.MOM(dataframe, timeperiod=10)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=5)
        
        # 布林带
        bollinger = ta.BBANDS(dataframe, timeperiod=20, nbdevup=2.0, nbdevdn=2.0)
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_lowerband'] = bollinger['lowerband']
        
        # ATR和波动率
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['volatility'] = dataframe['close'].rolling(window=20).std()
        
        # 成交量指标
        dataframe['volume_sma'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['volume_ratio'] = dataframe['volume'] / dataframe['volume_sma']
        
        # ===== 做多目标构建 =====
        
        # 1. 动量因子组合（标准化）
        momentum_rsi = (dataframe['rsi'] - 50) / 50  # 标准化到[-1,1]
        momentum_macd = np.tanh(dataframe['macdhist'] / dataframe['close'] * 1000)  # tanh标准化
        momentum_roc = np.tanh(dataframe['roc'] / 100)  # tanh标准化
        
        long_momentum = (momentum_rsi + momentum_macd + momentum_roc) / 3
        
        # 2. 趋势因子组合
        trend_ma = np.where(dataframe['close'] > dataframe['ma_20'], 1, -1) * \
                  np.where(dataframe['ma_20'] > dataframe['ma_50'], 1, 0.5) * \
                  np.where(dataframe['ma_50'] > dataframe['ma_100'], 1, 0.5)
        
        trend_position = (dataframe['close'] - dataframe['ma_20']) / dataframe['ma_20']
        trend_position = np.tanh(trend_position * 20)  # 标准化
        
        long_trend = (trend_ma + trend_position) / 2
        
        # 3. 波动率因子（逆向）
        bb_position = (dataframe['close'] - dataframe['bb_lowerband']) / \
                     (dataframe['bb_upperband'] - dataframe['bb_lowerband'])
        bb_position = np.clip(bb_position, 0, 1)  # 限制在[0,1]
        
        volatility_factor = 1 / (1 + dataframe['atr'] / dataframe['close'] * 100)  # 低波动率偏好
        
        long_volatility = (bb_position + volatility_factor) / 2
        
        # 做多目标合成（权重自动归一化）
        total_weight = self.momentum_weight.value + self.trend_weight.value + self.volatility_weight.value
        
        dataframe['target_long'] = (
            (self.momentum_weight.value / total_weight) * long_momentum +
            (self.trend_weight.value / total_weight) * long_trend +
            (self.volatility_weight.value / total_weight) * long_volatility
        )
        
        # ===== 做空目标构建 =====
        
        # 1. 做空动量因子（反向）
        short_momentum_rsi = (50 - dataframe['rsi']) / 50  # 反向RSI
        short_momentum_macd = -momentum_macd  # 反向MACD
        short_momentum_willr = (dataframe['willr'] + 50) / 50  # 威廉指标标准化
        
        short_momentum = (short_momentum_rsi + short_momentum_macd + short_momentum_willr) / 3
        
        # 2. 做空波动率因子（高波动偏好）
        high_volatility = dataframe['atr'] / dataframe['close'] * 100
        high_volatility = np.tanh(high_volatility)  # 标准化
        
        volume_spike = np.tanh((dataframe['volume_ratio'] - 1) * 2)  # 放量偏好
        
        short_volatility = (high_volatility + volume_spike) / 2
        
        # 3. 做空结构因子
        bear_structure = np.where(dataframe['close'] < dataframe['bb_lowerband'], 1, 0) * \
                        np.where(dataframe['close'] < dataframe['ma_20'], 1, 0.5) * \
                        np.where(dataframe['rsi'] < 30, 1.2, 1.0)  # 超卖加权
        
        resistance_break = np.where(
            (dataframe['close'] < dataframe['bb_middleband']) & 
            (dataframe['volume_ratio'] > 1.5), 1.2, 1.0
        )
        
        short_structure = bear_structure * resistance_break
        short_structure = np.tanh(short_structure - 1)  # 标准化到[-1,1]
        
        # 做空目标合成
        short_total_weight = (self.short_momentum_weight.value + 
                             self.short_volatility_weight.value + 
                             self.short_structure_weight.value)
        
        dataframe['target_short'] = (
            (self.short_momentum_weight.value / short_total_weight) * short_momentum +
            (self.short_volatility_weight.value / short_total_weight) * short_volatility +
            (self.short_structure_weight.value / short_total_weight) * short_structure
        )
        
        # ===== 市场环境过滤器 =====
        
        # 趋势强度过滤
        trend_strength = abs(dataframe['ma_20'] - dataframe['ma_50']) / dataframe['ma_50']
        trend_filter = np.where(trend_strength > 0.02, 1.0, 0.7)  # 强趋势时增强信号
        
        # 波动率过滤
        volatility_percentile = dataframe['volatility'].rolling(window=50).rank(pct=True)
        volatility_filter = np.where(
            volatility_percentile < 0.8, 1.0, 0.8  # 极高波动时降低信号
        )
        
        # 应用过滤器
        dataframe['target_long'] = dataframe['target_long'] * trend_filter * volatility_filter
        dataframe['target_short'] = dataframe['target_short'] * trend_filter * volatility_filter
        
        # ===== 设置FreqAI目标 =====
        dataframe['&-target_long'] = dataframe['target_long']
        dataframe['&-target_short'] = dataframe['target_short']
        dataframe['&-target'] = dataframe['target_long']  # 向后兼容
        
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        指标填充
        """
        self.freqai_info = self.config["freqai"]
        dataframe = self.freqai.start(dataframe, metadata, self)
        
        # 添加辅助指标用于信号确认
        dataframe['volume_ma'] = dataframe['volume'].rolling(window=20).mean()
        dataframe['price_change'] = dataframe['close'].pct_change()
        
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        入场信号 - 多空分离逻辑
        """
        # 做多条件
        enter_long_conditions = [
            df["do_predict"] == 1,
            df['&-target_long'] > self.threshold_long.value,
            df['volume'] > df['volume_ma'],  # 放量确认
            df['rsi'] > 35,  # 避免超卖时做多
            df['close'] > df['bb_lowerband'],  # 避免在下轨做多
        ]

        # 做空条件
        enter_short_conditions = [
            df["do_predict"] == 1,
            df['&-target_short'] > abs(self.threshold_short.value),  # 注意：做空目标为正值
            df['volume'] > df['volume_ma'] * 1.2,  # 更强的放量要求
            df['rsi'] < 65,  # 避免超买时做空
            df['close'] < df['bb_upperband'],  # 避免在上轨做空
        ]

        df.loc[
            reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
        ] = (1, "long_lstm2")

        df.loc[
            reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
        ] = (1, "short_lstm2")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        """
        出场信号 - 独立的退出逻辑
        """
        # 做多出场
        exit_long_conditions = [
            df["do_predict"] == 1,
            (
                (df['&-target_long'] < self.threshold_long_exit.value) |
                (df['rsi'] > 75) |  # 超买退出
                (df['close'] > df['bb_upperband'] * 1.02)  # 突破上轨过多退出
            )
        ]

        # 做空出场
        exit_short_conditions = [
            df["do_predict"] == 1,
            (
                (df['&-target_short'] < self.threshold_short_exit.value) |
                (df['rsi'] < 25) |  # 超卖退出
                (df['close'] < df['bb_lowerband'] * 0.98)  # 突破下轨过多退出
            )
        ]

        if exit_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_long_conditions), ["exit_long", "exit_tag"]
            ] = (1, "exit_long_lstm2")

        if exit_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_short_conditions), ["exit_short", "exit_tag"]
            ] = (1, "exit_short_lstm2")

        return df

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                           rate: float, time_in_force: str, current_time,
                           entry_tag, side: str, **kwargs) -> bool:
        """
        交易确认 - 最后的风险检查
        """
        # 获取最新数据
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return False
            
        last_candle = dataframe.iloc[-1]
        
        # 基础风险检查
        if last_candle['volume'] == 0:
            return False
            
        # 做多确认
        if side == 'long':
            # 确保不在极端超买状态
            if last_candle['rsi'] > 80:
                return False
            # 确保有足够的上涨空间
            if last_candle['close'] > last_candle['bb_upperband'] * 1.05:
                return False
                
        # 做空确认
        elif side == 'short':
            # 确保不在极端超卖状态
            if last_candle['rsi'] < 20:
                return False
            # 确保有足够的下跌空间
            if last_candle['close'] < last_candle['bb_lowerband'] * 0.95:
                return False
        
        return True

    def confirm_trade_exit(self, pair: str, trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time, **kwargs) -> bool:
        """
        退出确认 - 避免不必要的频繁交易
        """
        # 如果是止损或ROI，直接确认
        if exit_reason in ['stop_loss', 'roi', 'trailing_stop_loss']:
            return True
            
        # 对于信号退出，检查持仓时间
        if hasattr(trade, 'open_date_utc'):
            hold_time = (current_time - trade.open_date_utc).total_seconds() / 3600
            # 持仓少于2小时的交易需要更强的退出信号
            if hold_time < 2:
                dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
                if not dataframe.empty:
                    last_candle = dataframe.iloc[-1]
                    if trade.is_short:
                        return last_candle['&-target_short'] < 0.1
                    else:
                        return last_candle['&-target_long'] < 0.1
        
        return True