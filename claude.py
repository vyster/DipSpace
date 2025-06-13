import pandas as pd
import numpy as np
from datetime import datetime

# Load data (keeping your existing preprocessing)
nifty = pd.read_csv("Niftweekly.csv")
sensex = pd.read_csv("SensexWeekly.csv")

for df in [nifty, sensex]:
    df.rename(columns={'Change %': 'Change'}, inplace=True)
    df.drop(columns=['Open', 'High', 'Low'], inplace=True)
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
    df['Change'] = df['Change'].str.replace('%', '', regex=False).astype(float)
    df['Price'] = df['Price'].str.replace(',', '', regex=False).astype(float)
    df['Year'] = df['Date'].dt.year

print("Data loaded and preprocessed:")
print(f"Nifty shape: {nifty.shape}")
print(f"Sensex shape: {sensex.shape}")

# Configuration
FALL_THRESHOLD = -3  # Configurable threshold (change as needed)
MONTHLY_INVESTMENT = 10000
TOTAL_ANNUAL_BUDGET = 120000

def get_monthly_first_days(df, year):
    """Get first trading day of each month for a given year"""
    year_data = df[df['Year'] == year].copy()
    year_data = year_data.sort_values('Date')
    
    # Group by year-month and get first day
    year_data['YearMonth'] = year_data['Date'].dt.to_period('M')
    monthly_first = year_data.groupby('YearMonth').first().reset_index()
    
    return monthly_first

def strategy_1_regular_monthly(df, year):
    """Strategy 1: Regular monthly investment of 10k on first trading day"""
    monthly_data = get_monthly_first_days(df, year)
    
    investments = []
    total_units = 0
    total_invested = 0
    
    for _, row in monthly_data.iterrows():
        investment_amount = MONTHLY_INVESTMENT
        price = row['Price']
        units_bought = investment_amount / price
        
        total_units += units_bought
        total_invested += investment_amount
        
        investments.append({
            'Strategy': 'Regular Monthly',
            'Year': year,
            'Date': row['Date'],
            'Price': price,
            'Amount_Invested': investment_amount,
            'Units_Bought': units_bought,
            'Running_Total_Invested': total_invested,
            'Running_Total_Units': total_units
        })
    
    return investments, total_units, total_invested

def strategy_2_dip_buying(df, year, fall_threshold=FALL_THRESHOLD):
    """Strategy 2: Buy only during dips, with December fallback"""
    year_data = df[df['Year'] == year].copy()
    year_data = year_data.sort_values('Date')
    
    # Find weeks with falls greater than threshold
    dip_weeks = year_data[year_data['Change'] <= fall_threshold].copy()
    
    # Sort by change (worst falls first) and take up to 12
    dip_weeks = dip_weeks.sort_values('Change').head(12)
    
    investments = []
    total_units = 0
    total_invested = 0
    
    # Calculate investment per opportunity
    num_dips = len(dip_weeks)
    
    if num_dips > 0:
        if num_dips >= 12:
            # Have enough dips, invest equally
            investment_per_dip = TOTAL_ANNUAL_BUDGET / 12
            
            for _, row in dip_weeks.head(12).iterrows():
                price = row['Price']
                units_bought = investment_per_dip / price
                
                total_units += units_bought
                total_invested += investment_per_dip
                
                investments.append({
                    'Strategy': 'Dip Buying',
                    'Year': year,
                    'Date': row['Date'],
                    'Price': price,
                    'Amount_Invested': investment_per_dip,
                    'Units_Bought': units_bought,
                    'Weekly_Change': row['Change'],
                    'Running_Total_Invested': total_invested,
                    'Running_Total_Units': total_units
                })
        else:
            # Not enough dips, invest in available dips and remaining in December
            investment_per_dip = TOTAL_ANNUAL_BUDGET / 12
            
            # Invest in available dips
            for _, row in dip_weeks.iterrows():
                price = row['Price']
                units_bought = investment_per_dip / price
                
                total_units += units_bought
                total_invested += investment_per_dip
                
                investments.append({
                    'Strategy': 'Dip Buying',
                    'Year': year,
                    'Date': row['Date'],
                    'Price': price,
                    'Amount_Invested': investment_per_dip,
                    'Units_Bought': units_bought,
                    'Weekly_Change': row['Change'],
                    'Running_Total_Invested': total_invested,
                    'Running_Total_Units': total_units
                })
            
            # Invest remaining in last week of December
            remaining_investments = 12 - num_dips
            remaining_amount = remaining_investments * investment_per_dip
            
            # Find last trading day of December
            december_data = year_data[year_data['Date'].dt.month == 12]
            if not december_data.empty:
                last_december = december_data.iloc[-1]
                price = last_december['Price']
                units_bought = remaining_amount / price
                
                total_units += units_bought
                total_invested += remaining_amount
                
                investments.append({
                    'Strategy': 'Dip Buying (December Fallback)',
                    'Year': year,
                    'Date': last_december['Date'],
                    'Price': price,
                    'Amount_Invested': remaining_amount,
                    'Units_Bought': units_bought,
                    'Weekly_Change': last_december['Change'],
                    'Running_Total_Invested': total_invested,
                    'Running_Total_Units': total_units
                })
    
    return investments, total_units, total_invested

def calculate_current_value(total_units, current_price):
    """Calculate current value and returns"""
    current_value = total_units * current_price
    return current_value

def analyze_index(df, index_name, fall_threshold=FALL_THRESHOLD):
    """Complete analysis for an index"""
    
    print(f"\n{'='*60}")
    print(f"ANALYSIS FOR {index_name}")
    print(f"Fall Threshold: {fall_threshold}%")
    print(f"{'='*60}")
    
    # Get unique years
    years = sorted(df['Year'].unique())
    current_price = df.iloc[-1]['Price']  # Latest price as current price
    
    all_investments = []
    summary_results = []
    
    for year in years:
        print(f"\n--- Year {year} Analysis ---")
        
        # Strategy 1: Regular Monthly
        s1_investments, s1_units, s1_invested = strategy_1_regular_monthly(df, year)
        s1_current_value = calculate_current_value(s1_units, current_price)
        s1_return = ((s1_current_value - s1_invested) / s1_invested) * 100
        
        # Strategy 2: Dip Buying
        s2_investments, s2_units, s2_invested = strategy_2_dip_buying(df, year, fall_threshold)
        s2_current_value = calculate_current_value(s2_units, current_price)
        s2_return = ((s2_current_value - s2_invested) / s2_invested) * 100 if s2_invested > 0 else 0
        
        # Add current values to investment records
        for inv in s1_investments:
            inv['Current_Price'] = current_price
            inv['Current_Value_of_Units'] = inv['Units_Bought'] * current_price
            all_investments.append(inv)
        
        for inv in s2_investments:
            inv['Current_Price'] = current_price
            inv['Current_Value_of_Units'] = inv['Units_Bought'] * current_price
            all_investments.append(inv)
        
        # Summary for this year
        summary_results.append({
            'Index': index_name,
            'Year': year,
            'Strategy': 'Regular Monthly',
            'Total_Invested': s1_invested,
            'Total_Units': s1_units,
            'Current_Value': s1_current_value,
            'Absolute_Return': s1_current_value - s1_invested,
            'Return_Percentage': s1_return,
            'Number_of_Investments': len(s1_investments)
        })
        
        summary_results.append({
            'Index': index_name,
            'Year': year,
            'Strategy': 'Dip Buying',
            'Total_Invested': s2_invested,
            'Total_Units': s2_units,
            'Current_Value': s2_current_value,
            'Absolute_Return': s2_current_value - s2_invested,
            'Return_Percentage': s2_return,
            'Number_of_Investments': len(s2_investments)
        })
        
        # Print year summary
        print(f"Regular Monthly - Invested: ₹{s1_invested:,.0f}, Current Value: ₹{s1_current_value:,.0f}, Return: {s1_return:.2f}%")
        print(f"Dip Buying - Invested: ₹{s2_invested:,.0f}, Current Value: ₹{s2_current_value:,.0f}, Return: {s2_return:.2f}%")
        
        # Count actual dips for this year
        year_data = df[df['Year'] == year]
        dip_count = len(year_data[year_data['Change'] <= fall_threshold])
        print(f"Weeks with falls > {abs(fall_threshold)}%: {dip_count}")
    
    return all_investments, summary_results

# Main Analysis
print("Starting Investment Strategy Analysis...")

# Analyze both indices
nifty_investments, nifty_summary = analyze_index(nifty, "NIFTY 50", FALL_THRESHOLD)
sensex_investments, sensex_summary = analyze_index(sensex, "SENSEX", FALL_THRESHOLD)

# Combine all results
all_investments = nifty_investments + sensex_investments
all_summary = nifty_summary + sensex_summary

# Create comprehensive DataFrames
investments_df = pd.DataFrame(all_investments)
summary_df = pd.DataFrame(all_summary)

# Save to CSV files
investments_df.to_csv('detailed_investments.csv', index=False)
summary_df.to_csv('investment_summary.csv', index=False)

print(f"\n{'='*60}")
print("OVERALL SUMMARY")
print(f"{'='*60}")

# Display summary table
print("\nInvestment Summary by Index, Year, and Strategy:")
print(summary_df.to_string(index=False))

print(f"\nFiles saved:")
print(f"- detailed_investments.csv: Complete investment details")
print(f"- investment_summary.csv: Summary by year and strategy")

# Overall comparison
print(f"\n{'='*40}")
print("STRATEGY COMPARISON ACROSS ALL YEARS")
print(f"{'='*40}")

for index_name in ['NIFTY 50', 'SENSEX']:
    index_summary = summary_df[summary_df['Index'] == index_name]
    
    regular_total = index_summary[index_summary['Strategy'] == 'Regular Monthly']['Return_Percentage'].mean()
    dip_total = index_summary[index_summary['Strategy'] == 'Dip Buying']['Return_Percentage'].mean()
    
    print(f"\n{index_name}:")
    print(f"Average Return - Regular Monthly: {regular_total:.2f}%")
    print(f"Average Return - Dip Buying: {dip_total:.2f}%")
    
    if dip_total > regular_total:
        print(f"Winner: Dip Buying (Advantage: {dip_total - regular_total:.2f}%)")
    else:
        print(f"Winner: Regular Monthly (Advantage: {regular_total - dip_total:.2f}%)")

print(f"\n{'='*60}")
print("ANALYSIS COMPLETE!")
print(f"Fall threshold used: {FALL_THRESHOLD}%")
print("Change FALL_THRESHOLD variable to test different thresholds.")
print(f"{'='*60}")