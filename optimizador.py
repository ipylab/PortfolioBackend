
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from scipy.optimize import minimize

ticker_name_map = {
    # Criptomonedas
    "BTC-USD": ("Bitcoin", "cryptos"),
    "ETH-USD": ("Ethereum", "cryptos"),
    "BNB-USD": ("BNB", "cryptos"),
    "SOL-USD": ("Solana", "cryptos"),
    "XRP-USD": ("XRP", "cryptos"),

    # Commodities
    "GLD": ("Oro", "commodities"),
    "SLV": ("Plata", "commodities"),
    "NGF": ("Gas_Natural", "commodities"),
    "CLF": ("Petroleo_Crudo", "commodities"),
    "BZF": ("Brent_Oil", "commodities"),
    "WEAT": ("Trigo_ETF", "commodities"),
    "CORN": ("Maiz_ETF", "commodities"),
    "ZCF": ("Futuros_Maiz", "commodities"),
    "ZSF": ("Futuros_Soja", "commodities"),
    "ZWF": ("Futuros_Trigo", "commodities"),
    "PLF": ("Platino", "commodities"),
    "HGF": ("Cobre", "commodities"),

    # Sectores SPDR y ETFs temáticos
    "XLP": ("Consumo_Basico", "sectors"),
    "XLY": ("Consumo_Discrecional", "sectors"),
    "XLF": ("Financieros", "sectors"),
    "XLK": ("Tecnologia", "sectors"),
    "XLV": ("Salud", "sectors"),
    "XLI": ("Industriales", "sectors"),
    "XLE": ("Energia", "sectors"),
    "XLB": ("Materiales", "sectors"),
    "XLU": ("Utilities", "sectors"),
    "XLC": ("Comunicacion", "sectors"),
    "XLRE": ("Inmobiliario", "sectors"),
    "XBI": ("Biotecnologia_SmallCap", "sectors"),
    "IBB": ("Biotecnologia_IBB", "sectors"),
    "FIW": ("Infraestructura_Agua", "sectors"),
    "PHO": ("Recursos_Hidricos", "sectors"),
    "PAVE": ("Infraestructura", "sectors"),
    "PEJ": ("Viajes_Ocio", "sectors"),
    "HACK": ("Ciberseguridad", "sectors"),
    "BUG": ("Ciberamenazas", "sectors"),
    "BLOK": ("Blockchain", "sectors"),
    "ARKK": ("ARK_Innovation", "sectors"),
    "PBW": ("Energias_Limpias", "sectors"),
    "QCLN": ("CleanTech", "sectors"),
    "ICLN": ("Energia_Limpia_ICLN", "sectors"),
    "URA": ("Uranio", "sectors"),
    "REMX": ("Tierras_Raras", "sectors"),
    "ROBO": ("Robotica_IA", "sectors"),
    "BOTZ": ("Automatizacion", "sectors"),
    "EBIZ": ("E_Commerce", "sectors"),
    "FIVG": ("Redes_5G", "sectors"),
    "MOO": ("Agronegocio", "sectors"),
    "PAF": ("Asia_Pacifico", "sectors"),
    "SIF": ("Infraestructura_Sostenible", "sectors"),
    "KWEB": ("Internet_China", "sectors"),

    # Índices bursátiles globales
    "STOXX50E": ("Europa_Stoxx50", "economies"),
    "GSPC": ("SP500", "economies"),
    "GSPTSE": ("Canada_TSX", "economies"),
    "GDAXI": ("Alemania_DAX", "economies"),
    "FCHI": ("Francia_CAC40", "economies"),
    "FTSE": ("UK_FTSE100", "economies"),
    "IXIC": ("NASDAQ", "economies"),
    "RUT": ("Russell_2000", "economies"),
    "N225": ("Japon_Nikkei", "economies"),
    "BSESN": ("India_Sensex", "economies"),
    "BVSP": ("Brasil_Bovespa", "economies"),
    "AXJO": ("Australia_ASX200", "economies"),
    "KS11": ("Corea_KOSPI", "economies"),
    "000001.SS": ("China_Shanghai", "economies"),
    "TA125.TA": ("Israel_TA125", "economies")
}



name_to_ticker = {v[0]: k for k, v in ticker_name_map.items()}
risk_free_rate = 0.03

def load_prices(horizon_years, selected_asset_names):
    start_date = datetime.today() - timedelta(days=horizon_years * 365)
    base_url = "https://fcdiiuujwrwbvmzjxzkt.supabase.co/storage/v1/object/public/files/historicals/"
    prices = {}

    for name in selected_asset_names:
        ticker = name_to_ticker.get(name)
        if not ticker:
            continue
        display_name, category = ticker_name_map[ticker]
        url = f"{base_url}{category}/{ticker}.csv"
        try:
            response = requests.get(url)
            response.raise_for_status()
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            df.columns = [col.strip() for col in df.columns]
            df = df[["Date", "Adj Close"]] if "Adj Close" in df.columns else df.iloc[:, :2]
            df.columns = ["date", "price"]
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df = df.set_index("date").sort_index()
            df = df[df.index >= start_date].dropna()
            if not df.empty:
                prices[ticker] = df["price"]
        except Exception as e:
            print(f"⚠️ Error procesando {ticker}: {e}")
     

    if prices:
        print("✅ Activos cargados:")
        for ticker, serie in prices.items():
            print(f" - {ticker}: {len(serie)} registros desde {serie.index.min().date()}")
    else:
        print("⚠️ No se cargaron datos para ningún activo.")
    df = pd.DataFrame(prices)
    df = df.sort_index().ffill()
    df = df[df.notna().sum(axis=1) > 0]  # Elimina solo filas totalmente vacías
    df = df[df.index.dayofweek < 5]     # Mantiene solo días hábiles (lunes a viernes)
    return df

def compute_metrics(returns, weights, capital, benchmark=None, nivel_confianza=0.05):
    port_returns = returns @ weights
    daily_vol = np.std(port_returns)
    annual_vol = daily_vol * np.sqrt(252)
    mean_daily_return = port_returns.mean()
    annual_return = mean_daily_return * 252
    sharpe_ratio = (annual_return - risk_free_rate) / annual_vol

    downside_returns = port_returns[port_returns < 0]
    downside_dev = np.std(downside_returns)
    sortino_ratio = (annual_return - risk_free_rate) / (downside_dev * np.sqrt(252))

    cumulative = (port_returns).cumsum().apply(np.exp) * capital
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    max_drawdown = drawdown.min()
    calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else None

    total_return = cumulative.iloc[-1] / cumulative.iloc[0] - 1
    cagr = (1 + total_return) ** (252 / len(returns)) - 1

    # VaR y CVaR
    var = np.percentile(port_returns, 100 * nivel_confianza)
    cvar = port_returns[port_returns <= var].mean()

    # Tracking Error y Info Ratio (si hay benchmark)
    alpha = beta = tracking_error = info_ratio = corr = None
    if benchmark is not None:
        tracking_error = np.std(port_returns - benchmark)
        info_ratio = (annual_return - benchmark.mean() * 252) / tracking_error if tracking_error else None
        cov = np.cov(port_returns, benchmark)[0][1]
        beta = cov / np.var(benchmark)
        alpha = annual_return - beta * benchmark.mean() * 252
        corr = np.corrcoef(port_returns, benchmark)[0][1]

    # Herfindahl-Hirschman Index (HHI)
    hhi = np.sum(weights ** 2)
    diversification_ratio = (np.dot(weights, np.std(returns, axis=0))) / np.std(port_returns)

    # Historial
    historico = [{"fecha": str(f), "valor": float(v)} for f, v in cumulative.items()]

    return {
        "rentabilidad": round(annual_return, 4),
        "volatilidad": round(annual_vol, 4),
        "sharpe": round(sharpe_ratio, 4),
        "sortino": round(sortino_ratio, 4),
        "drawdown_max": round(max_drawdown, 4),
        "calmar_ratio": round(calmar_ratio, 4) if calmar_ratio else None,
        "retorno_total": round(total_return, 4),
        "retorno_anualizado": round(cagr, 4),
        "var": round(var, 4),
        "cvar": round(cvar, 4),
        "tracking_error": round(tracking_error, 4) if tracking_error else None,
        "info_ratio": round(info_ratio, 4) if info_ratio else None,
        "alpha": round(alpha, 4) if alpha else None,
        "beta": round(beta, 4) if beta else None,
        "correlacion_benchmark": round(corr, 4) if corr else None,
        "diversificacion_hhi": round(hhi, 4),
        "diversificacion_ratio": round(diversification_ratio, 4),
        "historico": historico
    }

def risk_parity_weights(cov_matrix):
    inv_vol = 1 / np.sqrt(np.diag(cov_matrix))
    weights = inv_vol / np.sum(inv_vol)
    return weights

def optimizar_portafolio(activos, capital, horizonte, metodo, peso_max=1.0, pesos_manual=None):
    # Agregar GSPC internamente solo como benchmark si el usuario no lo pidió
    usar_gspc_como_benchmark = "S&P 500" not in activos
    activos_con_benchmark = activos + ["S&P 500"] if usar_gspc_como_benchmark else activos

    df = load_prices(horizonte, activos_con_benchmark)
    if df.empty:
        return {"error": "No hay datos disponibles."}

    returns = np.log(df / df.shift(1)).fillna(0)
    activos_validos = list(df.columns)

    # GSPC como benchmark
    benchmark = returns["GSPC"] if "GSPC" in df.columns else None

    # Excluir GSPC del portafolio si no fue solicitado explícitamente
    activos_optimizables = [a for a in activos_validos if a != "GSPC" or "S&P 500" in activos]

    # Validación extra
    if not activos_optimizables:
        return {"error": "Ningún activo válido disponible para optimización."}

    mean_returns = returns[activos_optimizables].mean() * 252
    cov_matrix = returns[activos_optimizables].cov() * 252

    # Cálculo de pesos según método
    if metodo == "Equal Weight":
        weights = np.ones(len(activos_optimizables)) / len(activos_optimizables)

    elif metodo == "Risk Parity":
        weights = risk_parity_weights(cov_matrix)

    elif metodo == "Manual":
        if not pesos_manual or len(pesos_manual) != len(activos_optimizables):
            return {"error": "Pesos manuales inválidos o mal dimensionados"}
        weights = np.array(pesos_manual)

    else:
        def sharpe(w):
            r = np.dot(w, mean_returns)
            v = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
            return -(r - risk_free_rate) / v

        init_guess = len(activos_optimizables) * [1. / len(activos_optimizables)]
        bounds = tuple((0, peso_max) for _ in activos_optimizables)
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        result = minimize(sharpe, init_guess, bounds=bounds, constraints=constraints)
        weights = result.x

    weights = weights / np.sum(weights)

    portafolio = [
        {
            "activo": name,
            "peso": round(float(w), 4),
            "asignacion": round(float(w * capital), 2)
        }
        for name, w in zip(activos_optimizables, weights) if w > 0.01
    ]

    metrics = compute_metrics(returns[activos_optimizables], weights, capital, benchmark)

    return {
        "portafolio": portafolio,
        **metrics
    }