from fastapi import APIRouter, Depends
from luma.api.schemas import TeacherRequest, TeacherResponse
from luma.api.controllers.chat_controller import get_luma_service
from luma.api.services.luma_service import LumaService

router = APIRouter()

@router.post("/teacher/start", response_model=TeacherResponse)
async def start_teacher(
    request: TeacherRequest,
    service: LumaService = Depends(get_luma_service),
) -> TeacherResponse:
    session = await service.start_teacher_mode(request.user_id, request.topic)
    return TeacherResponse(
        session_id=session.session_id,
        status=session.status,
        lessons=session.lessons,
        explanations=session.explanations,
        exercises=session.exercises,
    )

@router.post("/teacher/continue", response_model=TeacherResponse)
async def continue_teacher(
    request: TeacherRequest,
    service: LumaService = Depends(get_luma_service),
) -> TeacherResponse:
    session = await service.continue_teacher_mode(request.user_id, request.topic)
    return TeacherResponse(
        session_id=session.session_id,
        status=session.status,
        lessons=session.lessons,
        explanations=session.explanations,
        exercises=session.exercises,
    )
