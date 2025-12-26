import os

BASE_URL = "https://raw.githubusercontent.com/symfony/symfony-docs/7.3/"
FILES = [
    "doctrine.rst", "routing.rst", "security.rst", "validation.rst",
    "service_container.rst", "forms.rst", "translation.rst",
    "configuration.rst", "http_cache.rst", "cache.rst",
    "serializer.rst", "messenger.rst", "console.rst",
    "mailer.rst", "event_dispatcher.rst", "bundles.rst",
    "best_practices.rst", "performance.rst", "index.rst","rate_limiter.rst",
    "workflow.rst","webhook.rst","web_link.rst","scheduler.rst","setup/docker.rst",
    "setup/symfony_cli.rst","setup/web_server_configuration.rst","components/cache.rst",
    "components/dependency_injection.rst","components/form.rst","components/dependency_injection/compilation.rst",
    "components/cache/cache_invalidation.rst","components/cache/cache_items.rst","components/var_exporter.rst",
    "components/runtime.rst","components/property_access.rst","configuration/secrets.rst","console/calling_commands.rst",
    "console/coloring.rst","console/command_in_controller.rst","console/commands_as_services.rst","console/lazy_commands.rst","console/style.rst",
    
]

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
INDEX_DIR = "index"

MODEL_NAME = "all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
