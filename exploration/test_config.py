from crip_x.utils.config import settings, ROOT_DIR

print(f"Root dir: {ROOT_DIR}")
print(f"Environment: {settings.environment}")
print(f"API will run on: {settings.api_host}:{settings.api_port}")
print(f"Signal window: {settings.signal_window_seconds}s")
print(f"Reliability threshold: {settings.reliability_threshold}")