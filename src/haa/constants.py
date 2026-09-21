ASSETS = ("SPY", "TIP", "IEF", "BIL")
LEVERAGED_ASSETS = ("SSO",)
# IEF belongs to Classic HAA's eight-asset risk-on ranking as well as its
# defensive pair; BIL is defensive-only. QQQ is intentionally absent.
CLASSIC_OFFENSIVE_ASSETS = ("IEF", "SPY", "IWM", "PDBC", "TLT", "VEA", "VNQ", "VWO")
CLASSIC_DEFENSIVE_ASSETS = ("IEF", "BIL")
CLASSIC_DATA_ASSETS = ("TIP", "BIL", *CLASSIC_OFFENSIVE_ASSETS)
STRATEGY_NAME = "HAA-Simple"
MOMENTUM_LOOKBACKS = (1, 3, 6, 12)
DEFAULT_TAX_RATE = 0.25
