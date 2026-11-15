"""What a VLM must answer, and how an answer becomes retrieval queries.

The schema goes to Ollama as a JSON schema, so the model can only write answers of that
shape. Every reply is still validated, and one that fails is recorded as a failed image.
"""

from pydantic import BaseModel, field_validator


class TagsV1(BaseModel):
    """A flat list of short noun phrases."""

    tags: list[str]

    @field_validator("tags")
    @classmethod
    def _clean(cls, tags: list[str]) -> list[str]:
        cleaned = list(dict.fromkeys(tag.strip() for tag in tags if tag.strip()))
        if not cleaned:
            raise ValueError("the answer has no tags")
        return cleaned

    def queries(self) -> list[tuple[str, float]]:
        """The texts to look up, each with a weight. Every tag counts the same."""
        return [(tag, 1.0) for tag in self.tags]


SCHEMAS = {"tags_v1": TagsV1}


def answer_type(schema: str) -> type[TagsV1]:
    """The model an answer is validated against. Replies under "none" become TagsV1."""
    return TagsV1 if schema == "none" else SCHEMAS[schema]


def json_schema(schema: str) -> dict | None:
    """What Ollama enforces while the model writes its reply. Nothing under "none"."""
    return None if schema == "none" else SCHEMAS[schema].model_json_schema()


def parse(reply: str, schema: str) -> TagsV1:
    """Validates a reply. Under "none" it is read as a comma-separated list."""
    if schema == "none":
        return TagsV1(tags=reply.split(","))
    return SCHEMAS[schema].model_validate_json(reply)
