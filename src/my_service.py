# Imports required by the service's model
import pandas as pd
from lazypredict import LazyClassifier
from sklearn.model_selection import train_test_split
import io

api_description = """This service benchmarks a dataset with various models and outputs the results sorted by accuracy.
In order for the service to work your dataset label column must be called "target".
Also to improve the results you may want to remove unneccessary columns from the dataset.
Finally, avoid having multiple empty lines at the end of the file.
"""
api_summary = """This service benchmarks a dataset with various models and outputs the results sorted by accuracy.
"""
api_title = "Classification benchmark API."
version = "0.0.1"

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
