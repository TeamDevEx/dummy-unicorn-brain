import os 
class Config:
    cached = {
        "V1_0_URL": '/1.0',
        "NP_V1_0_URL": '/1.0',
        "PR_V1_0_URL": '/1.0',
        "V2_0_URL": '/2.0',
        "NP_V2_0_URL": '/2.0',
        "PR_V2_0_URL": '/2.0',
    }
    env = f"{os.environ.get('API_ENV')}-" if os.environ.get("API_ENV") else ""

    if "API_GSM" in os.environ:
        from google.cloud import secretmanager
        backend = "gsm"
        gsm_project = os.environ.get("API_GCP_PROJ")
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
        except:
            pass
        if value is not None:
            return cls.cache(key, value)
        elif default is not None:
            return cls.cache(key, default)
        else:
            raise ValueError(f"{cls.gsm_project}/secrets/{key.replace('_', '-').lower()}/versions/latest not found")

    @classmethod
    def cache(cls, key, value):
        if value is not None:
            cls.cached[key] = value
        return value
