from fastapi import APIRouter, Depends
from luma.api.schemas import ChatRequest, ChatResponse
from luma.api.services.luma_service import LumaService

router = APIRouter()

# Application-scoped LumaService instance (set by routes.py at startup)
_luma_service: LumaService | None = None

def get_luma_service() -> LumaService:
    """Dependency provider returning the application-scoped LumaService."""
    if _luma_service is None:
        raise RuntimeError("LumaService has not been initialized")
    return _luma_service

def set_luma_service(service: LumaService) -> None:
    """Set the application-scoped LumaService instance."""
    global _luma_service
    _luma_service = service

@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service: LumaService = Depends(get_luma_service),
) -> ChatResponse:
    result = await service.process_chat(request.user_id, request.message)
    return ChatResponse(**result)
