from fastapi import APIRouter, Depends, Query
from luma.api.schemas import PersonalizationResponse
from luma.api.controllers.chat_controller import get_luma_service
from luma.api.services.luma_service import LumaService

router = APIRouter()

@router.get("/personalization", response_model=PersonalizationResponse)
async def get_personalization(
    user_id: str = Query(min_length=1),
    service: LumaService = Depends(get_luma_service),
) -> PersonalizationResponse:
    ctx = await service.get_personalization(user_id)
    return PersonalizationResponse(
        tone=ctx.tone,
        style=ctx.style,
        focus=ctx.focus,
        reasons=ctx.reasons,    
    )
