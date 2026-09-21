ASSETS = ("SPY", "TIP", "IEF", "BIL")
LEVERAGED_ASSETS = ("SSO",)
# IEF belongs to Classic HAA's eight-asset risk-on ranking as well as its
# defensive pair; BIL is defensive-only. QQQ is intentionally absent.
CLASSIC_OFFENSIVE_ASSETS = ("IEF", "SPY", "IWM", "PDBC", "TLT", "VEA", "VNQ", "VWO")
CLASSIC_DEFENSIVE_ASSETS = ("IEF", "BIL")
CLASSIC_DATA_ASSETS = ("TIP", "BIL", *CLASSIC_OFFENSIVE_ASSETS)
# HAA's leveraged flavour calculates every momentum score on these 1x
# underlyings, then maps the selected holding to its 2x substitute. PDBC and
# BIL remain unleveraged because no substitute is part of this flavour.
CLASSIC_LEVERAGED_SUBSTITUTIONS = {
    "IEF": "UST",
    "SPY": "SSO",
    "IWM": "UWM",
    "PDBC": "PDBC",
    "TLT": "UBT",
    "VEA": "EFO",
    "VNQ": "URE",
    "VWO": "EET",
    "BIL": "BIL",
}
CLASSIC_LEVERAGED_HOLDINGS = ("UST", "SSO", "UWM", "PDBC", "UBT", "EFO", "URE", "EET", "BIL")
CLASSIC_LEVERAGED_DATA_ASSETS = tuple(dict.fromkeys((*CLASSIC_DATA_ASSETS, *CLASSIC_LEVERAGED_HOLDINGS)))
STRATEGY_NAME = "HAA-Simple"
MOMENTUM_LOOKBACKS = (1, 3, 6, 12)
DEFAULT_TAX_RATE = 0.25
