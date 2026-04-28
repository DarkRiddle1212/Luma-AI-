"""
API Layer

Handles HTTP requests and responses, delegates to service layer.
"""


def __getattr__(name):
    if name == "router":
        from luma.api.routes import router
        return router
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["router"]
