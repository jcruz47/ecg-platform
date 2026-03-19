from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    supabase_url: str
    supabase_service_role_key: str
    supabase_publishable_key: str

    signal_bucket: str = "ecg-signals-raw"
    image_bucket: str = "ecg-images-raw"
    report_bucket: str = "ecg-reports"
    derived_bucket: str = "ecg-derived"

    default_sampling_rate: int = 360

    use_llm: bool = True
    lm_studio_base_url: str = "http://127.0.0.1:1234/v1"
    lm_studio_model: str = "meta-llama-3.1-8b-instruct"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()