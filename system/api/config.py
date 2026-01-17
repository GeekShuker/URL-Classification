from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    redis_url: str
    app_workdir: str = "/app"
    log_dir: str = "/app/reports/system_logs"

settings = Settings()
