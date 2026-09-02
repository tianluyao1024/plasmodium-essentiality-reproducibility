param([string]$Config = "config/formal_paths.yaml", [string]$Stage = "all")
python run_reproduction.py --config $Config --stage $Stage
