import sys
import json
import importlib.util
import traceback


def main():
    strategy_path, params_json, data_dir, split, cost_bps = sys.argv[1:6]

    import engine  # the copy sitting right next to this file in the sandbox

    data = engine.load_split(data_dir, split)

    spec = importlib.util.spec_from_file_location("strategy", strategy_path)
    strategy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy)

    params = json.loads(params_json)
    weights = strategy.target_weights(data, **params)

    bt = engine.backtest(weights, data["returns"], cost_bps=float(cost_bps))
    scored = bt.loc[data["eval_start"]:]

    print(json.dumps({
        "ok": True,
        "metrics": engine.metrics(scored),
        "equity": [round(float(x), 6) for x in (1 + scored["ret"]).cumprod().tolist()],
        "dates": [str(x.date()) for x in scored.index],
    }))



if __name__ == "__main__":
    try:
        main()
    except Exception:
        print(json.dumps({"ok": False, "error": traceback.format_exc(limit=3)}))
