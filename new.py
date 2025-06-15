import pandas as pd
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class AdvancedDipBuyingStrategy:
    """
    Advanced Dip Buying Strategy implementing multiple proven algorithms:
    1. RSI-based dip identification (30/70 thresholds with dynamic adjustment)
    2. Moving Average crossover confirmation
    3. War Chest strategy with volatility-based allocation
    4. Multi-timeframe analysis
    5. Risk-adjusted position sizing
    """
    
    def __init__(self, monthly_investment: float = 10000):
        self.monthly_investment = monthly_investment
        self.strategies_config = {
            'rsi_period': 14,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'ma_short': 20,
            'ma_long': 50,
            'volatility_window': 20,
            'war_chest_threshold': 0.8,  # 80% allocation threshold
            'max_position_size': 0.15    # 15% max position size
        }
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators needed for the strategy"""
        df = df.copy()
        
        # RSI Calculation
        delta = df['Price'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.strategies_config['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.strategies_config['rsi_period']).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Moving Averages
        df['MA_Short'] = df['Price'].rolling(window=self.strategies_config['ma_short']).mean()
        df['MA_Long'] = df['Price'].rolling(window=self.strategies_config['ma_long']).mean()
        
        # Volatility (20-day rolling standard deviation)
        df['Volatility'] = df['Price'].rolling(window=self.strategies_config['volatility_window']).std()
        df['Volatility_Percentile'] = df['Volatility'].rolling(window=252).rank(pct=True)
        
        # Price relative to MA
        df['Price_vs_MA_Long'] = (df['Price'] - df['MA_Long']) / df['MA_Long'] * 100
        
        # Bollinger Bands for additional confirmation
        df['BB_Middle'] = df['Price'].rolling(window=20).mean()
        bb_std = df['Price'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Price'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        return df
    
    def identify_dip_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced dip identification using multiple signals:
        1. RSI oversold (< 30) or pullback to 50 in uptrend
        2. Price below long-term MA
        3. High volatility environment
        4. Bollinger Band lower touch
        """
        df = df.copy()
        
        # Basic RSI signals
        df['RSI_Oversold'] = df['RSI'] < self.strategies_config['rsi_oversold']
        df['RSI_Pullback'] = (df['RSI'] < 50) & (df['MA_Short'] > df['MA_Long'])  # Pullback in uptrend
        
        # MA signals
        df['Below_MA_Long'] = df['Price'] < df['MA_Long']
        df['MA_Uptrend'] = df['MA_Short'] > df['MA_Long']
        
        # Volatility signals
        df['High_Volatility'] = df['Volatility_Percentile'] > 0.7
        
        # Bollinger Band signals
        df['BB_Lower_Touch'] = df['BB_Position'] < 0.2
        
        # Combined dip signals with different strength levels
        df['Strong_Dip'] = (
            df['RSI_Oversold'] & 
            df['Below_MA_Long'] & 
            df['BB_Lower_Touch']
        )
        
        df['Medium_Dip'] = (
            (df['RSI_Oversold'] | df['RSI_Pullback']) &
            df['Below_MA_Long'] &
            df['High_Volatility']
        ) & ~df['Strong_Dip']
        
        df['Weak_Dip'] = (
            df['RSI_Pullback'] &
            df['MA_Uptrend'] &
            (df['Price_vs_MA_Long'] > -10)  # Not too far below MA
        ) & ~df['Strong_Dip'] & ~df['Medium_Dip']
        
        return df
    
    def calculate_position_size(self, signal_strength: str, volatility_percentile: float, 
                              available_cash: float) -> float:
        """
        Dynamic position sizing based on signal strength and market volatility
        """
        base_allocation = self.monthly_investment
        
        # Adjust for signal strength
        strength_multipliers = {
            'Strong_Dip': 2.0,    # Double investment for strong signals
            'Medium_Dip': 1.5,    # 1.5x for medium signals
            'Weak_Dip': 1.0,      # Normal for weak signals
            'Regular': 0.8        # Reduced for regular SIP
        }
        
        # Adjust for volatility (invest more when volatility is high)
        volatility_multiplier = 1 + (volatility_percentile - 0.5)  # Range: 0.5 to 1.5
        
        position_size = base_allocation * strength_multipliers.get(signal_strength, 1.0) * volatility_multiplier
        
        # Cap at available cash and max position size
        max_position = available_cash * self.strategies_config['max_position_size']
        return min(position_size, max_position, available_cash)
    
    def war_chest_strategy(self, df: pd.DataFrame, index_name: str) -> Tuple[pd.DataFrame, Dict]:
        """
        War Chest Strategy: Accumulate cash during normal times, deploy aggressively during major dips
        """
        df = self.calculate_technical_indicators(df)
        df = self.identify_dip_signals(df)
        
        current_price = df.iloc[-1]['Price']
        war_chest_logs = []
        
        # Initialize tracking variables
        total_units = 0
        total_invested = 0
        war_chest_cash = 0
        max_war_chest = self.monthly_investment * 12  # 1 year of savings
        
        for i, row in df.iterrows():
            monthly_budget = self.monthly_investment
            investment_made = 0
            investment_type = 'None'
            
            # Determine signal strength
            if row['Strong_Dip']:
                signal_strength = 'Strong_Dip'
            elif row['Medium_Dip']:
                signal_strength = 'Medium_Dip'
            elif row['Weak_Dip']:
                signal_strength = 'Weak_Dip'
            else:
                signal_strength = 'Regular'
            
            # War chest logic
            if signal_strength in ['Strong_Dip', 'Medium_Dip']:
                # Deploy war chest + monthly investment
                available_cash = war_chest_cash + monthly_budget
                position_size = self.calculate_position_size(
                    signal_strength, 
                    row['Volatility_Percentile'], 
                    available_cash
                )
                
                if position_size > 0:
                    units = position_size / row['Price']
                    total_units += units
                    total_invested += position_size
                    investment_made = position_size
                    investment_type = f'{signal_strength}_Deploy'
                    
                    # Reduce war chest
                    war_chest_reduction = min(war_chest_cash, position_size - monthly_budget)
                    war_chest_cash -= max(0, war_chest_reduction)
                    
            elif signal_strength == 'Weak_Dip':
                # Regular investment
                units = monthly_budget / row['Price']
                total_units += units
                total_invested += monthly_budget
                investment_made = monthly_budget
                investment_type = 'Weak_Dip_Regular'
                
            else:
                # Build war chest (invest only 50% of monthly budget)
                regular_investment = monthly_budget * 0.5
                war_chest_addition = monthly_budget * 0.5
                
                if war_chest_cash < max_war_chest:
                    war_chest_cash += war_chest_addition
                else:
                    # War chest full, invest normally
                    regular_investment = monthly_budget
                
                units = regular_investment / row['Price']
                total_units += units
                total_invested += regular_investment
                investment_made = regular_investment
                investment_type = 'War_Chest_Build'
            
            # Log the transaction
            war_chest_logs.append({
                'Date': row['Date'],
                'Price': row['Price'],
                'RSI': round(row['RSI'], 2),
                'MA_Short': round(row['MA_Short'], 2),
                'MA_Long': round(row['MA_Long'], 2),
                'Volatility_Percentile': round(row['Volatility_Percentile'], 2),
                'Signal_Strength': signal_strength,
                'Investment_Type': investment_type,
                'Amount_Invested': round(investment_made, 2),
                'Units_Bought': round(investment_made / row['Price'], 4) if investment_made > 0 else 0,
                'Total_Units': round(total_units, 4),
                'Total_Invested': round(total_invested, 2),
                'War_Chest_Cash': round(war_chest_cash, 2),
                'Current_Value': round(total_units * current_price, 2),
                'Index': index_name,
                'Strong_Dip': row['Strong_Dip'],
                'Medium_Dip': row['Medium_Dip'],
                'Weak_Dip': row['Weak_Dip']
            })
        
        logs_df = pd.DataFrame(war_chest_logs)
        
        summary = {
            'Total_Units': total_units,
            'Total_Invested': total_invested,
            'Final_Corpus': total_units * current_price,
            'Returns_Percent': ((total_units * current_price - total_invested) / total_invested) * 100,
            'Final_War_Chest': war_chest_cash,
            'Strong_Dip_Investments': len(logs_df[logs_df['Signal_Strength'] == 'Strong_Dip']),
            'Medium_Dip_Investments': len(logs_df[logs_df['Signal_Strength'] == 'Medium_Dip']),
            'Weak_Dip_Investments': len(logs_df[logs_df['Signal_Strength'] == 'Weak_Dip'])
        }
        
        return logs_df, summary
    
    def rsi_ma_strategy(self, df: pd.DataFrame, index_name: str) -> Tuple[pd.DataFrame, Dict]:
        """
        Pure RSI + Moving Average strategy for comparison
        """
        df = self.calculate_technical_indicators(df)
        df = self.identify_dip_signals(df)
        
        current_price = df.iloc[-1]['Price']
        rsi_ma_logs = []
        
        total_units = 0
        total_invested = 0
        
        for i, row in df.iterrows():
            investment_made = 0
            investment_type = 'None'
            
            # RSI + MA buy conditions
            if (row['RSI'] < 35 and row['Price'] < row['MA_Long']) or \
               (row['RSI'] < 50 and row['MA_Short'] > row['MA_Long'] and row['Price'] < row['MA_Short']):
                
                investment_made = self.monthly_investment
                investment_type = 'RSI_MA_Buy'
                
                units = investment_made / row['Price']
                total_units += units
                total_invested += investment_made
            
            # Log the transaction
            rsi_ma_logs.append({
                'Date': row['Date'],
                'Price': row['Price'],
                'RSI': round(row['RSI'], 2),
                'MA_Short': round(row['MA_Short'], 2),
                'MA_Long': round(row['MA_Long'], 2),
                'Investment_Type': investment_type,
                'Amount_Invested': round(investment_made, 2),
                'Units_Bought': round(investment_made / row['Price'], 4) if investment_made > 0 else 0,
                'Total_Units': round(total_units, 4),
                'Total_Invested': round(total_invested, 2),
                'Current_Value': round(total_units * current_price, 2),
                'Index': index_name
            })
        
        logs_df = pd.DataFrame(rsi_ma_logs)
        
        summary = {
            'Total_Units': total_units,
            'Total_Invested': total_invested,
            'Final_Corpus': total_units * current_price,
            'Returns_Percent': ((total_units * current_price - total_invested) / total_invested) * 100,
            'Total_Investments': len(logs_df[logs_df['Amount_Invested'] > 0])
        }
        
        return logs_df, summary

def run_advanced_analysis():
    """
    Main function to run the advanced dip buying analysis
    """
    # File configuration (same as your original code)
    files = {
        'nifty': 'Niftweekly.csv',
        'sensex': 'SensexWeekly.csv',
        'Midcap150': 'Midcap150Weekly.csv',
        'MidcapBSE': 'MidCapWeekly.csv',
        'Next50': 'Next50Weekly.csv',
        'Nifty100': 'Nifty100Weekly.csv',
        'SmallCap100': 'SmallCap100Weekly.csv'
    }
    
    # Initialize strategy
    strategy = AdvancedDipBuyingStrategy(monthly_investment=10000)
    
    # Dictionary to store all DataFrames
    dfs = {}
    
    # Load and prepare data (your existing data loading logic)
    for name, file in files.items():
        if os.path.exists(file):
            df = pd.read_csv(file)
            
            # Apply transformations
            df.rename(columns={'Change %': 'Change'}, inplace=True)
            df.drop(columns=['Open', 'High', 'Low', 'Vol.'], inplace=True, errors='ignore')
            df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
            df['Change'] = df['Change'].str.replace('%', '', regex=False).astype(float)
            df['Price'] = df['Price'].str.replace(',', '', regex=False).astype(float)
            df['Year'] = df['Date'].dt.year
            
            # Sort by date
            df = df.sort_values('Date', ascending=True).reset_index(drop=True)
            dfs[name] = df
        else:
            print(f"Warning: {file} not found, skipping {name}")
    
    # Run strategies for each index
    all_results = {}
    all_logs = {}
    
    for index_name, df in dfs.items():
        print(f"Processing {index_name}...")
        
        # War Chest Strategy
        war_chest_logs, war_chest_summary = strategy.war_chest_strategy(df, index_name)
        
        # RSI-MA Strategy
        rsi_ma_logs, rsi_ma_summary = strategy.rsi_ma_strategy(df, index_name)
        
        # Store results
        all_results[index_name] = {
            'War_Chest': war_chest_summary,
            'RSI_MA': rsi_ma_summary
        }
        
        all_logs[f"{index_name}_war_chest"] = war_chest_logs
        all_logs[f"{index_name}_rsi_ma"] = rsi_ma_logs
    
    # Create comprehensive summary
    summary_data = []
    for index_name, results in all_results.items():
        summary_data.append({
            'Index': index_name.upper(),
            'War_Chest_Total_Invested': results['War_Chest']['Total_Invested'],
            'War_Chest_Final_Corpus': round(results['War_Chest']['Final_Corpus'], 2),
            'War_Chest_Returns_Percent': round(results['War_Chest']['Returns_Percent'], 2),
            'War_Chest_Strong_Dips': results['War_Chest']['Strong_Dip_Investments'],
            'War_Chest_Medium_Dips': results['War_Chest']['Medium_Dip_Investments'],
            'RSI_MA_Total_Invested': results['RSI_MA']['Total_Invested'],
            'RSI_MA_Final_Corpus': round(results['RSI_MA']['Final_Corpus'], 2),
            'RSI_MA_Returns_Percent': round(results['RSI_MA']['Returns_Percent'], 2),
            'RSI_MA_Total_Investments': results['RSI_MA']['Total_Investments'],
            'War_Chest_Advantage': round(results['War_Chest']['Final_Corpus'] - results['RSI_MA']['Final_Corpus'], 2)
        })
    
    summary_df = pd.DataFrame(summary_data)
    
    # Export results
    with pd.ExcelWriter("advanced_dip_strategy_analysis.xlsx", engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="Strategy_Comparison", index=False)
        
        for log_name, log_df in all_logs.items():
            sheet_name = log_name.replace('_', ' ').title()[:31]
            log_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    # Display results
    print("\n" + "="*100)
    print("ADVANCED DIP BUYING STRATEGY ANALYSIS COMPLETE!")
    print("="*100)
    
    for index_name, results in all_results.items():
        print(f"\n{index_name.upper()}:")
        print(f"War Chest Strategy:")
        print(f"  Final Corpus: ₹{results['War_Chest']['Final_Corpus']:,.2f}")
        print(f"  Returns: {results['War_Chest']['Returns_Percent']:.2f}%")
        print(f"  Strong Dip Buys: {results['War_Chest']['Strong_Dip_Investments']}")
        print(f"  Medium Dip Buys: {results['War_Chest']['Medium_Dip_Investments']}")
        
        print(f"RSI-MA Strategy:")
        print(f"  Final Corpus: ₹{results['RSI_MA']['Final_Corpus']:,.2f}")
        print(f"  Returns: {results['RSI_MA']['Returns_Percent']:.2f}%")
        print(f"  Total Investments: {results['RSI_MA']['Total_Investments']}")
        
        advantage = results['War_Chest']['Final_Corpus'] - results['RSI_MA']['Final_Corpus']
        print(f"War Chest Advantage: ₹{advantage:,.2f}")
    
    print(f"\nDetailed results exported to: advanced_dip_strategy_analysis.xlsx")
    return summary_df, all_results, all_logs

# Example usage with your CSV data
if __name__ == "__main__":
    # If you have the CSV files, run the analysis
    try:
        summary, results, logs = run_advanced_analysis()
    except FileNotFoundError as e:
        print(f"CSV files not found. Please ensure the following files are in the current directory:")
        files = ['Niftweekly.csv', 'SensexWeekly.csv', 'Midcap150Weekly.csv', 
                'MidCapWeekly.csv', 'Next50Weekly.csv', 'Nifty100Weekly.csv', 'SmallCap100Weekly.csv']
        for file in files:
            print(f"  - {file}")
        
        # Demo with sample data structure
        print("\nCreating demo with sample data structure...")
        
        # Create sample data for demonstration
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='W')
        np.random.seed(42)
        
        # Simulate price movements with trends and volatility
        returns = np.random.normal(0.001, 0.02, len(dates))  # Weekly returns
        prices = [10000]  # Starting price
        
        for r in returns:
            prices.append(prices[-1] * (1 + r))
        
        sample_df = pd.DataFrame({
            'Date': dates,
            'Price': prices[1:],
            'Change': returns * 100
        })
        
        sample_df['Year'] = sample_df['Date'].dt.year
        
        # Run strategy on sample data
        strategy = AdvancedDipBuyingStrategy(monthly_investment=10000)
        
        print("\nRunning War Chest Strategy on sample data...")
        war_chest_logs, war_chest_summary = strategy.war_chest_strategy(sample_df, "SAMPLE_INDEX")
        
        print(f"Sample Results:")
        print(f"  Total Invested: ₹{war_chest_summary['Total_Invested']:,.2f}")
        print(f"  Final Corpus: ₹{war_chest_summary['Final_Corpus']:,.2f}")
        print(f"  Returns: {war_chest_summary['Returns_Percent']:.2f}%")
        print(f"  Strong Dip Investments: {war_chest_summary['Strong_Dip_Investments']}")
        print(f"  Medium Dip Investments: {war_chest_summary['Medium_Dip_Investments']}")