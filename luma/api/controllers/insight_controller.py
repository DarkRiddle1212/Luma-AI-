from typing import Optional
from fastapi import APIRouter, Depends
from luma.api.schemas import InsightResponse, InsightMomentsResponse
from luma.api.controllers.chat_controller import get_luma_service
from luma.api.services.luma_service import LumaService

router = APIRouter()

@router.get("/insights", response_model=InsightResponse)
async def get_insights(
    namespace: Optional[str] = None,
    service: LumaService = Depends(get_luma_service),
) -> InsightResponse:
    insights = await service.get_insights(namespace=namespace)
    return InsightResponse(insights=insights)

@router.get("/insight-moments", response_model=InsightMomentsResponse)
async def get_insight_moments(
    service: LumaService = Depends(get_luma_service),
) -> InsightMomentsResponse:
    moments = await service.get_insight_moments()
    return InsightMomentsResponse(insight_moments=moments)
