from datetime import datetime, timezone
from typing import Optional, List, Any, Dict

# Pydantic v2 运行约定：
# - 当前项目已明确依赖 Pydantic 2.x（见 requirements.txt），因此这里的实现目标是 v2 专用逻辑。
# - 这不是为了兼容 v1，而是为了保证代码在当前环境中使用官方、稳定且最现代的 API。
# - 传统的 v1 兼容分支在本项目中没有实际价值；保留它仅作为历史参考，不作为当前执行路径。
# - 也就是说：当前的真实语义是“v2-first、v1-fallback 仅用于参考”，而不是“继续维护 v1 兼容”。
try:
    # Pydantic v2：当前项目使用的正式 API，包含 ConfigDict / field_serializer 等现代能力。
    # 这些 API 是 v2 的标准接口，适用于 schema 定制和 JSON 序列化。
    from pydantic import ConfigDict, BaseModel, Field, field_serializer
    PYDANTIC_V2 = True
except ImportError:
    # 历史兼容保留：
    # - 这个分支只代表旧版 Pydantic 的回退代码。
    # - 在当前项目中它不会被命中，因为依赖已经固定为 pydantic==2.10.6。
    # - 保留注释而不删除，方便未来对比和排查“为什么这里还有 v1 代码”。
    from pydantic import BaseModel, Field
    ConfigDict = None
    field_serializer = None
    PYDANTIC_V2 = False

from modules import sd_samplers
from modules.api.models import (
    StableDiffusionTxt2ImgProcessingAPI,
    StableDiffusionImg2ImgProcessingAPI,
)


# 用于把 datetime 转成 ISO-8601 + Z 的字符串格式，适合 API 输出。
def convert_datetime_to_iso_8601_with_z_suffix(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z" if dt else None


# 确保所有时间字段都被统一转换到 UTC，避免跨时区造成不一致。
def transform_to_utc_datetime(dt: datetime) -> datetime:
    return dt.astimezone(tz=timezone.utc)


# 移除 OpenAPI/JSON Schema 中无意义的字段，减少前端和 API 文档噪音。
def strip_schema_extra(schema: Dict[str, Any]) -> None:
    props = schema.get("properties", {})
    props.pop("send_images", None)
    props.pop("save_images", None)

# API 查询参数：分页参数，供 /queue /history 这类接口使用。
class QueueStatusAPI(BaseModel):
    limit: Optional[int] = Field(title="Limit", description="The maximum number of tasks to return", default=20)
    offset: Optional[int] = Field(title="Offset", description="The offset of the tasks to return", default=0)


# 任务模型：
# - 这是整个 Agent Scheduler 的核心数据结构。
# - 任务状态、参数、结果和时间字段都在这里落地。
# - 它必须在不同 Pydantic 版本下都能被 FastAPI / Gradio 正确解析，否则 API 返回和内部数据模型都会异常。
class TaskModel(BaseModel):
    id: str = Field(title="Task Id")
    api_task_id: Optional[str] = Field(title="API Task Id", default=None)
    api_task_callback: Optional[str] = Field(title="API Task Callback", default=None)
    name: Optional[str] = Field(None, title="Task Name")
    type: str = Field(title="Task Type", description="Either txt2img or img2img")
    status: str = Field(
        "pending",
        title="Task Status",
        description="Either pending, running, done or failed",
    )
    params: Dict[str, Any] = Field(title="Task Parameters", description="The parameters of the task in JSON format")
    priority: Optional[int] = Field(None, title="Task Priority")
    position: Optional[int] = Field(None, title="Task Position")
    result: Optional[str] = Field(None, title="Task Result", description="The result of the task in JSON format")
    bookmarked: Optional[bool] = Field(None, title="Is task bookmarked")
    created_at: Optional[datetime] = Field(
        title="Task Created At",
        description="The time when the task was created",
        default=None,
    )
    updated_at: Optional[datetime] = Field(
        title="Task Updated At",
        description="The time when the task was updated",
        default=None,
    )

    # Pydantic v2：使用 field_serializer 让 JSON 输出的 datetime 统一转成毫秒时间戳。
    # 这条逻辑的意义在于：API 需要稳定、可比较的时间格式，而不是 Python 的原始 datetime 对象。
    if PYDANTIC_V2:
        @field_serializer('created_at', "updated_at", when_used="json")
        def serialize_datetime(self, dt: datetime) -> int:
            return int(dt.timestamp() * 1e3)

    # Pydantic v1：使用 Config.json_encoders 做等效兼容。
    # 这意味着同样的语义在 v1 下也能达到一致的序列化结果，保证 API 行为不受版本差异影响。
    class Config:
        json_encoders = {datetime: lambda dt: int(dt.timestamp() * 1e3)}


# txt2img 任务参数扩展：
# - 继承 SD WebUI 的原生 API 参数，保持和上游接口一致。
# - 同时新增 Agent Scheduler 自己的 checkpoint / vae / callback 字段。
# - 这一步的意义是：让扩展既能复用上游参数结构，又能加自己的业务字段，而不会破坏原始 API 兼容性。
class Txt2ImgApiTaskArgs(StableDiffusionTxt2ImgProcessingAPI):
    checkpoint: Optional[str] = Field(
        None,
        title="Custom checkpoint.",
        description="Custom checkpoint hash. If not specified, the latest checkpoint will be used.",
    )
    vae: Optional[str] = Field(
        None,
        title="Custom VAE.",
        description="Custom VAE. If not specified, the current VAE will be used.",
    )
    sampler_index: Optional[str] = Field(sd_samplers.samplers[0].name, title="Sampler name", alias="sampler_name")
    callback_url: Optional[str] = Field(
        None,
        title="Callback URL",
        description="The callback URL to send the result to.",
    )

    # 这里的兼容分支的意义：
    # - v2 通过 model_config 调整 schema；
    # - v1 通过 Config.schema_extra 达到同样目标；
    # - 目的是让 OpenAPI / 自动生成的前端文档中少掉不需要暴露的字段，避免接口被污染。
    if PYDANTIC_V2:
        model_config = ConfigDict(json_schema_extra=strip_schema_extra)
    else:
        class Config:
            schema_extra = staticmethod(strip_schema_extra)


# img2img 任务参数扩展：
# - 与 txt2img 同结构，只是对应 image-to-image 的参数模型。
# - 这里也保留兼容层，是因为 img2img 参数同样依赖上游 SD WebUI 参数类，以及 Pydantic 版本差异。
class Img2ImgApiTaskArgs(StableDiffusionImg2ImgProcessingAPI):
    checkpoint: Optional[str] = Field(
        None,
        title="Custom checkpoint.",
        description="Custom checkpoint hash. If not specified, the latest checkpoint will be used.",
    )
    vae: Optional[str] = Field(
        None,
        title="Custom VAE.",
        description="Custom VAE. If not specified, the current VAE will be used.",
    )
    sampler_index: Optional[str] = Field(sd_samplers.samplers[0].name, title="Sampler name", alias="sampler_name")
    callback_url: Optional[str] = Field(
        None,
        title="Callback URL",
        description="The callback URL to send the result to.",
    )

    # Pydantic v2：通过 model_config 调整 schema，去掉不需要暴露的字段。
    if PYDANTIC_V2:
        model_config = ConfigDict(json_schema_extra=strip_schema_extra)
    else:
        # Pydantic v1：使用 Config.schema_extra 做等效处理。
        class Config:
            schema_extra = staticmethod(strip_schema_extra)


# 队列任务创建后的返回模型，返回新创建任务的 id。
class QueueTaskResponse(BaseModel):
    task_id: str = Field(title="Task Id")


# 当前队列状态返回模型：
# - current_task_id: 当前正在执行的任务
# - pending_tasks: 等待中的任务列表
# - total_pending_tasks: 总队列长度
class QueueStatusResponse(BaseModel):
    current_task_id: Optional[str] = Field(None, title="Current Task Id", description="The on progress task id")
    pending_tasks: List[TaskModel] = Field(title="Pending Tasks", description="The pending tasks in the queue")
    total_pending_tasks: int = Field(title="Queue length", description="The total pending tasks in the queue")
    paused: bool = Field(title="Paused", description="Whether the queue is paused")


# 历史任务返回模型：
# - tasks: 历史任务列表
# - total: 总记录数
class HistoryResponse(BaseModel):
    tasks: List[TaskModel] = Field(title="Tasks")
    total: int = Field(title="Task count")


# 任务更新请求参数：
# 允许修改任务名、checkpoint、params 等字段。
class UpdateTaskArgs(BaseModel):
    name: Optional[str] = Field(None, title="Task Name")
    checkpoint: Optional[str] = None
    params: Optional[Dict[str, Any]] = Field(
        None, title="Task Parameters", description="The parameters of the task in JSON format"
    )
