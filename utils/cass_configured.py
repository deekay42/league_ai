
import time
from cassiopeia import get_default_config, set_default_region, apply_settings, Item, Patch, Tier, Division, Side, \
    get_challenger_league, LeagueEntries, get_grandmaster_league, get_master_league, Queue, Lane, get_match
from cassiopeia.core.match import EventData

config = get_default_config()
# config['pipeline']['ChampionGG'] = {
#     "package": "cassiopeia_championgg",
#     "api_key": "496b4c2f287421a51a41aeb51a808f74"  # Your api.champion.gg API key (or an env var containing it)
# }

config['pipeline']['RiotAPI'] = {
    "api_key": "RGAPI-8aa60c8e-77c5-4ea8-9de4-1646faefc3a9",
    "limiting_share": 1.0,
    "request_error_handling": {
        "404": {
            "strategy": "throw"
        },
        "429": {
            "service": {
                "strategy": "exponential_backoff",
                "initial_backoff": 1.0,
                "backoff_factor": 2.0,
                "max_attempts": 6
            },
            "method": {
                "strategy": "retry_from_headers",
                "max_attempts": 6
            },
            "application": {
                "strategy": "retry_from_headers",
                "max_attempts": 6
            }
        },
        "500": {
            "strategy": "exponential_backoff",
            "initial_backoff": 1.0,
            "backoff_factor": 2.0,
            "max_attempts": 6
        },
        "503": {
            "strategy": "exponential_backoff",
            "initial_backoff": 1.0,
            "backoff_factor": 2.0,
            "max_attempts": 6
        },
        "timeout": {
            "strategy": "exponential_backoff",
            "initial_backoff": 1.0,
            "backoff_factor": 2.0,
            "max_attempts": 6
        },
        "403": {
            "strategy": "throw"
        }
    }
}

set_default_region("EUW")
apply_settings(config)
