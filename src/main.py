import asyncio
import time
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from common_code.config import get_settings
from common_code.http_client import HttpClient
from common_code.logger.logger import get_logger, Logger
from common_code.service.controller import router as service_router
from common_code.service.service import ServiceService
from common_code.storage.service import StorageService
from common_code.tasks.controller import router as tasks_router
from common_code.tasks.service import TasksService
from common_code.tasks.models import TaskData
from common_code.service.models import Service
from common_code.service.enums import ServiceStatus
from common_code.common.enums import (
    FieldDescriptionType,
    ExecutionUnitTagName,
    ExecutionUnitTagAcronym,
)
from common_code.common.models import FieldDescription, ExecutionUnitTag
from contextlib import asynccontextmanager

# Imports required by the service's model
import pandas as pd
from lazypredict import LazyClassifier
from sklearn.model_selection import train_test_split
import io

settings = get_settings()


class MyService(Service):
    """
    Benchmark multiple models on a dataset and return the results.
    """

    # Any additional fields must be excluded for Pydantic to work
    _model: object
    _logger: Logger

    def __init__(self):
        super().__init__(
            name="Classification Benchmark",
            slug="classification-benchmark",
            url=settings.service_url,
            summary=api_summary,
            description=api_description,
            status=ServiceStatus.AVAILABLE,
            data_in_fields=[
                FieldDescription(name="dataset", type=[FieldDescriptionType.TEXT_CSV]),
            ],
            data_out_fields=[
                FieldDescription(name="result", type=[FieldDescriptionType.TEXT_PLAIN]),
            ],
            tags=[
                ExecutionUnitTag(
                    name=ExecutionUnitTagName.DATA_PREPROCESSING,
                    acronym=ExecutionUnitTagAcronym.DATA_PREPROCESSING,
                ),
            ],
            has_ai=False,
            docs_url="https://docs.swiss-ai-center.ch/reference/services/classification-benchmark/",
        )
        self._logger = get_logger(settings)

    def process(self, data):
        raw = str(data["dataset"].data.decode("utf-8-sig").encode("utf-8"))
        raw = (
            raw.replace(",", ";")
            .replace("\\n", "\n")
            .replace("\\r", "\n")
            .replace("b'", "")
        )

        lines = raw.splitlines()
        if lines[-1] == "" or lines[-1] == "'":
            lines.pop()
        raw = "\n".join(lines)

        data_df = pd.read_csv(io.StringIO(raw), sep=";")

        X = data_df.drop("target", axis=1)
        y = data_df["target"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        clf = LazyClassifier(verbose=0, custom_metric=None)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)

        buf = io.BytesIO()
        buf.write(models.to_string().encode("utf-8"))

        return {
            "result": TaskData(
                data=buf.getvalue(), type=FieldDescriptionType.TEXT_PLAIN
            ),
        }


service_service: ServiceService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Manual instances because startup events doesn't support Dependency Injection
    # https://github.com/tiangolo/fastapi/issues/2057
    # https://github.com/tiangolo/fastapi/issues/425

    # Global variable
    global service_service

    # Startup
    logger = get_logger(settings)
    http_client = HttpClient()
    storage_service = StorageService(logger)
    my_service = MyService()
    tasks_service = TasksService(logger, settings, http_client, storage_service)
    service_service = ServiceService(logger, settings, http_client, tasks_service)

    tasks_service.set_service(my_service)

    # Start the tasks service
    tasks_service.start()

    async def announce():
        retries = settings.engine_announce_retries
        for engine_url in settings.engine_urls:
            announced = False
            while not announced and retries > 0:
                announced = await service_service.announce_service(
                    my_service, engine_url
                )
                retries -= 1
                if not announced:
                    time.sleep(settings.engine_announce_retry_delay)
                    if retries == 0:
                        logger.warning(
                            f"Aborting service announcement after "
                            f"{settings.engine_announce_retries} retries"
                        )

    # Announce the service to its engine
    asyncio.ensure_future(announce())

    yield

    # Shutdown
    for engine_url in settings.engine_urls:
        await service_service.graceful_shutdown(my_service, engine_url)


api_description = """This service benchmarks a dataset with various models and outputs the results sorted by accuracy.
In order for the service to work your dataset label column must be called "target".
Also to improve the results you may want to remove unneccessary columns from the dataset.
Finally, avoid having multiple empty lines at the end of the file.
"""
api_summary = """This service benchmarks a dataset with various models and outputs the results sorted by accuracy.
"""

# Define the FastAPI application with information
app = FastAPI(
    lifespan=lifespan,
    title="Classification benchmark API.",
    description=api_description,
    version="0.0.1",
    contact={
        "name": "Swiss AI Center",
        "url": "https://swiss-ai-center.ch/",
        "email": "info@swiss-ai-center.ch",
    },
    swagger_ui_parameters={
        "tagsSorter": "alpha",
        "operationsSorter": "method",
    },
    license_info={
        "name": "GNU Affero General Public License v3.0 (GNU AGPLv3)",
        "url": "https://choosealicense.com/licenses/agpl-3.0/",
    },
)

# Include routers from other files
app.include_router(service_router, tags=["Service"])
app.include_router(tasks_router, tags=["Tasks"])

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Redirect to docs
@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse("/docs", status_code=301)
