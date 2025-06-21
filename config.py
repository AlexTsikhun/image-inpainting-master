from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    FILE_ID_DATA: str
    FILE_ID_INDEX: str
    PORT: int

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

config = Config()
