
def get_model(model_name):
    if model_name == "ridge":
        try:
            from sklearn.linear_model import Ridge
            model = Ridge(alpha=1.0)
        except ImportError:
            raise ImportError("Please install scikit-learn: `pip install scikit-learn`")
        
    elif model_name == "xgboost":
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                objective="reg:squarederror",
                random_state=42,
                verbosity=0
            )
        except ImportError:
            raise ImportError("Please install XGBoost: `pip install xgboost`")
        
    elif model_name == "random_forest":
        try:
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=16,
                random_state=42,
                n_jobs=-1
            )
        except ImportError:
            raise ImportError("Please install scikit-learn: `pip install scikit-learn`")
        
    elif model_name == "tabpfn":
        try:
            from tabpfn import TabPFNRegressor
        except ImportError:
            raise ImportError("Please install tabpfn: `pip install tabpfn`")
        model = TabPFNRegressor()  # no hyperparameters needed
        
    elif model_name == "tabstar":
        try:
            from tabstar.tabstar_model import TabSTARRegressor
        except ImportError:
            raise ImportError("Please install TabSTAR via `pip install tabstar`")
        model = TabSTARRegressor()
        
    else:
        raise ValueError(f"Model {model_name} is not supported.")
    return model
