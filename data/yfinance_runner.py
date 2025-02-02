import yfinance as yf
import pandas as pd

def fetch_data(symbol) -> pd.DataFrame:
    symbol = symbol + ".T"
    # 全期間のデータを取得
    data = yf.Ticker(symbol).history(period="max")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.droplevel(1)
    return data

def fetch_additional_info(symbol):
    symbol = symbol + ".T"
    ticker = yf.Ticker(symbol)
    
    # ファンダメンタルデータの取得
    financials = ticker.financials
    balance_sheet = ticker.balance_sheet
    cashflow = ticker.cashflow

    # アナリストの推奨
    recommendations = ticker.recommendations

    return financials, balance_sheet, cashflow, recommendations

def main():
    symbol = "7203"  # トヨタ自動車のシンボル
    data = fetch_data(symbol)
    print("株価データ:")
    print(data)

    financials, balance_sheet, cashflow, recommendations = fetch_additional_info(symbol)
    print("\nファンダメンタルデータ:")
    print(financials)
    print("\nバランスシート:")
    print(balance_sheet)
    print("\nキャッシュフロー:")
    print(cashflow)
    print("\nアナリストの推奨:")
    print(recommendations)

if __name__ == "__main__":
    main()
