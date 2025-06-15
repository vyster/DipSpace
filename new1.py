
import pandas as pd
import os
import numpy as np
from scipy.optimize import fsolve
from datetime import datetime

# Helper functions for CAGR and XIRR calculations
def calculate_cagr(beginning_value, ending_value, years):
    """Calculate Compound Annual Growth Rate"""
    if beginning_value <= 0 or ending_value <= 0 or years <= 0:
        return 0
    return ((ending_value / beginning_value) ** (1/years) - 1) * 100

def xirr(cash_flows, dates, guess=0.1):
    """Calculate XIRR (Extended Internal Rate of Return)"""
    if len(cash_flows) != len(dates):
        return None
    
    if len(cash_flows) < 2:
        return None
    
    def npv(rate):
        """Net Present Value calculation for XIRR"""
        base_date = min(dates)
        return sum([cf / (1 + rate) ** ((date - base_date).days / 365.0) 
                   for cf, date in zip(cash_flows, dates)])
    
    try:
        result = fsolve(npv, guess)[0]
        return result * 100  # Convert to percentage
    except:
        return None
files = {
    'nifty': 'Niftweekly.csv',
    'sensex': 'SensexWeekly.csv',
    'Midcap150': 'Midcap150Weekly.csv',
    'MidcapBSE': 'MidCapWeekly.csv',
    'Next50': 'Next50Weekly.csv',
    'Nifty100': 'Nifty100Weekly.csv',
    'SmallCap100': 'SmallCap100Weekly.csv'
}

# Verify files exist
missing_files = []
for name, file in files.items():
    if not os.path.exists(file):
        missing_files.append(file)

if missing_files:
    raise FileNotFoundError("Some required files are missing!")

# Dictionary to store all DataFrames
dfs = {}

# Load all data files
for name, file in files.items():
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
    
    # Store in dictionary
    dfs[name] = df

# Function to simulate SIP strategy
def simulate_sip_strategy(df, index_name, monthly_investment=10000):
    years = df['Year'].unique()
    current_price = df.iloc[-1]['Price']
    last_date = df.iloc[-1]['Date']
    sip_logs = []
    total_units = 0
    total_invested = 0
    
    # For XIRR calculation
    cash_flows = []
    dates = []

    for year in sorted(years):
        for month in range(1, 13):
            month_data = df[(df['Year'] == year) & (df['Date'].dt.month == month)]
            if not month_data.empty:
                # Use first week of the month for SIP
                row = month_data.iloc[0]
                units = monthly_investment / row['Price']
                total_units += units
                total_invested += monthly_investment
                
                # Add cash flow (negative for investment)
                cash_flows.append(-monthly_investment)
                dates.append(row['Date'])
                
                sip_logs.append({
                    'Strategy': 'SIP',
                    'Date': row['Date'],
                    'Price': row['Price'],
                    'Amount Invested': monthly_investment,
                    'Units': units,
                    'Total Units': total_units,
                    'Total Invested': total_invested,
                    'Current Value': total_units * current_price,
                    'Index': index_name,
                    'Year': year,
                    'Month': month
                })

    # Add final value as positive cash flow for XIRR
    final_value = total_units * current_price
    cash_flows.append(final_value)
    dates.append(last_date)
    
    # Calculate CAGR
    start_date = min([d for d in dates[:-1]])  # Exclude final date
    end_date = last_date
    years_invested = (end_date - start_date).days / 365.25
    cagr = calculate_cagr(total_invested, final_value, years_invested)
    
    # Calculate XIRR
    xirr_rate = xirr(cash_flows, dates)

    return pd.DataFrame(sip_logs), total_units, total_invested, final_value, cagr, xirr_rate

# Function to simulate DIP strategy
def simulate_dip_strategy(df, index_name, fall_threshold=-3, monthly_investment=10000):
    years = df['Year'].unique()
    current_price = df.iloc[-1]['Price']
    last_date = df.iloc[-1]['Date']
    dip_logs = []
    total_units = 0
    total_invested = 0
    
    # For XIRR calculation
    cash_flows = []
    dates = []

    for year in sorted(years):
        # Find dip opportunities in the year
        dip_data = df[(df['Year'] == year) & (df['Change'] < fall_threshold)].copy()
        monthly_investments_made = 0
        
        # Invest in dips (up to 12 times per year)
        for _, row in dip_data.iterrows():
            if monthly_investments_made >= 12:
                break
                
            units = monthly_investment / row['Price']
            total_units += units
            total_invested += monthly_investment
            monthly_investments_made += 1
            
            # Add cash flow (negative for investment)
            cash_flows.append(-monthly_investment)
            dates.append(row['Date'])
            
            dip_logs.append({
                'Strategy': 'DIP',
                'Date': row['Date'],
                'Price': row['Price'],
                'Amount Invested': monthly_investment,
                'Units': units,
                'Total Units': total_units,
                'Total Invested': total_invested,
                'Current Value': total_units * current_price,
                'Index': index_name,
                'Year': year,
                'Investment_Type': 'Dip_Buy'
            })
        
        # If less than 12 investments made, fill remaining with December investments
        if monthly_investments_made < 12:
            remaining_investments = 12 - monthly_investments_made
            dec_data = df[(df['Year'] == year) & (df['Date'].dt.month == 12)]
            
            if not dec_data.empty:
                # Use last available price in December
                dec_price = dec_data.iloc[-1]['Price']
                dec_date = dec_data.iloc[-1]['Date']
                
                for i in range(remaining_investments):
                    units = monthly_investment / dec_price
                    total_units += units
                    total_invested += monthly_investment
                    
                    # Add cash flow (negative for investment)
                    cash_flows.append(-monthly_investment)
                    dates.append(dec_date)
                    
                    dip_logs.append({
                        'Strategy': 'DIP',
                        'Date': dec_date,
                        'Price': dec_price,
                        'Amount Invested': monthly_investment,
                        'Units': units,
                        'Total Units': total_units,
                        'Total Invested': total_invested,
                        'Current Value': total_units * current_price,
                        'Index': index_name,
                        'Year': year,
                        'Investment_Type': 'Dec_Fill'
                    })

    # Add final value as positive cash flow for XIRR
    final_value = total_units * current_price
    cash_flows.append(final_value)
    dates.append(last_date)
    
    # Calculate CAGR
    start_date = min([d for d in dates[:-1]])  # Exclude final date
    end_date = last_date
    years_invested = (end_date - start_date).days / 365.25
    cagr = calculate_cagr(total_invested, final_value, years_invested)
    
    # Calculate XIRR
    xirr_rate = xirr(cash_flows, dates)

    return pd.DataFrame(dip_logs), total_units, total_invested, final_value, cagr, xirr_rate

# Dictionary to store all results
results = {}
logs = {}

# Process each DataFrame
fall_threshold = -3
monthly_investment = 10000

for index_name, df in dfs.items():
    # Generate SIP strategy results
    sip_logs, sip_total_units, sip_total_invested, sip_final_corpus, sip_cagr, sip_xirr = simulate_sip_strategy(
        df, index_name, monthly_investment
    )
    
    # Generate DIP strategy results
    dip_logs, dip_total_units, dip_total_invested, dip_final_corpus, dip_cagr, dip_xirr = simulate_dip_strategy(
        df, index_name, fall_threshold, monthly_investment
    )
    
    # Store logs
    logs[f"{index_name}_sip"] = sip_logs
    logs[f"{index_name}_dip"] = dip_logs
    
    # Store summary results
    results[index_name] = {
        'SIP': {
            'Total Units': sip_total_units,
            'Total Invested': sip_total_invested,
            'Final Corpus': sip_final_corpus,
            'Returns': ((sip_final_corpus - sip_total_invested) / sip_total_invested) * 100,
            'CAGR': sip_cagr,
            'XIRR': sip_xirr
        },
        'DIP': {
            'Total Units': dip_total_units,
            'Total Invested': dip_total_invested,
            'Final Corpus': dip_final_corpus,
            'Returns': ((dip_final_corpus - dip_total_invested) / dip_total_invested) * 100,
            'CAGR': dip_cagr,
            'XIRR': dip_xirr
        }
    }

# Create summary DataFrame
summary_data = []
for index_name, data in results.items():
    summary_data.append({
        'Index': index_name.upper(),
        'SIP_Total_Units': round(data['SIP']['Total Units'], 2),
        'SIP_Total_Invested': data['SIP']['Total Invested'],
        'SIP_Final_Corpus': round(data['SIP']['Final Corpus'], 2),
        'SIP_Returns_Percent': round(data['SIP']['Returns'], 2),
        'SIP_CAGR_Percent': round(data['SIP']['CAGR'], 2) if data['SIP']['CAGR'] else 0,
        'SIP_XIRR_Percent': round(data['SIP']['XIRR'], 2) if data['SIP']['XIRR'] else 0,
        'DIP_Total_Units': round(data['DIP']['Total Units'], 2),
        'DIP_Total_Invested': data['DIP']['Total Invested'],
        'DIP_Final_Corpus': round(data['DIP']['Final Corpus'], 2),
        'DIP_Returns_Percent': round(data['DIP']['Returns'], 2),
        'DIP_CAGR_Percent': round(data['DIP']['CAGR'], 2) if data['DIP']['CAGR'] else 0,
        'DIP_XIRR_Percent': round(data['DIP']['XIRR'], 2) if data['DIP']['XIRR'] else 0,
        'DIP_vs_SIP_Corpus_Advantage': round(data['DIP']['Final Corpus'] - data['SIP']['Final Corpus'], 2),
        'DIP_vs_SIP_XIRR_Advantage': round((data['DIP']['XIRR'] or 0) - (data['SIP']['XIRR'] or 0), 2)
    })

summary_df = pd.DataFrame(summary_data)

# Export all results to Excel
with pd.ExcelWriter("dip_vs_sip_analysis_corrected.xlsx", engine="openpyxl") as writer:
    # Write summary sheet
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    
    # Write individual strategy logs
    for log_name, log_df in logs.items():
        sheet_name = log_name.replace('_', ' ').title()[:31]  # Excel sheet name limit
        log_df.to_excel(writer, sheet_name=sheet_name, index=False)

# Display final summary
print("Analysis Complete!")
print("\nSUMMARY RESULTS:")
print("=" * 100)
for index_name, data in results.items():
    print(f"\n{index_name.upper()}:")
    print(f"SIP Strategy:")
    print(f"  Units: {data['SIP']['Total Units']:.2f} | "
          f"Corpus: ₹{data['SIP']['Final Corpus']:,.2f} | "
          f"Returns: {data['SIP']['Returns']:.2f}%")
    print(f"  CAGR: {data['SIP']['CAGR']:.2f}% | "
          f"XIRR: {data['SIP']['XIRR']:.2f}%" if data['SIP']['XIRR'] else "  CAGR: N/A | XIRR: N/A")
    
    print(f"DIP Strategy:")
    print(f"  Units: {data['DIP']['Total Units']:.2f} | "
          f"Corpus: ₹{data['DIP']['Final Corpus']:,.2f} | "
          f"Returns: {data['DIP']['Returns']:.2f}%")
    print(f"  CAGR: {data['DIP']['CAGR']:.2f}% | "
          f"XIRR: {data['DIP']['XIRR']:.2f}%" if data['DIP']['XIRR'] else "  CAGR: N/A | XIRR: N/A")
    
    corpus_advantage = data['DIP']['Final Corpus'] - data['SIP']['Final Corpus']
    xirr_advantage = (data['DIP']['XIRR'] or 0) - (data['SIP']['XIRR'] or 0)
    print(f"DIP Advantage: ₹{corpus_advantage:,.2f} | XIRR Advantage: {xirr_advantage:.2f}%")

print(f"\nResults exported to: dip_vs_sip_analysis_corrected.xlsx")