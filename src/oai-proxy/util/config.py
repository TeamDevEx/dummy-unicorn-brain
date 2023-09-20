import os
class Config:
    cached = {}
    env = f"{os.environ.get('OAI_PROXY_ENV')}-" if os.environ.get("OAI_PROXY_ENV") else ""

    if "OAI_PROXY_USE_GSM" in os.environ:
        from google.cloud import secretmanager
        backend = "gsm"
        gsm_project = os.environ.get("OAI_PROXY_GSM_PROJ")
        client = secretmanager.SecretManagerServiceClient()
    else:
        from dotenv import load_dotenv
        load_dotenv()
        backend = "env"


    @classmethod
    def fetch(cls, key, cache=True, default=None):
        key = f"{cls.env}{key}".replace('-', '_').upper()
        if cache and key in cls.cached:
            return cls.cached[key]
        else:
            return getattr(cls, f"fetch_{cls.backend}")(key, default)


    @classmethod
    def fetch_env(cls, key, default):
        if key in os.environ:
            return cls.cache(key, os.environ.get(key))
        elif default is not None:
            return cls.cache(key, default)
        else:
            raise ValueError(f"{key} not in ENV or .env file")


    @classmethod
    def fetch_gsm(cls, key, default):
        value = None
        location = f"projects/{cls.gsm_project}/secrets/{key.replace('_', '-').lower()}/versions/latest"
        try:
            response = cls.client.access_secret_version(request={"name": location})
            value = response.payload.data.decode("UTF-8")
        except Exception as e:
            print(f"Unable to fetch {location}", e)
            pass
        if value is not None:
            return cls.cache(key, value)
        elif default is not None:
            return cls.cache(key, default)
        else:
            raise ValueError(f"{location} not found")

    @classmethod
    def cache(cls, key, value):
        if value is not None:
            cls.cached[key] = value
        return value

    @classmethod
    def clear_cache(cls):
        cls.cached = {}
