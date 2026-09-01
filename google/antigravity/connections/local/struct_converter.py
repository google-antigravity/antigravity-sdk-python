# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Structured data conversion utilities for LocalConnection and genai.Content."""

from collections import abc
import dataclasses
from typing import Any, cast

from google.protobuf import message

from google.antigravity.proto import content_pb2

# Automatically discover all Content and media component Message classes from
# content.proto:
_CONTENT_PROTO_TYPES: tuple[type[Any], ...] = tuple(
    v
    for v in content_pb2.__dict__.values()
    if isinstance(v, type)
    and issubclass(v, message.Message)
    and getattr(v, "DESCRIPTOR", None) is not None
    and v.DESCRIPTOR.file.name.endswith("content.proto")
    and (
        v.DESCRIPTOR.name.endswith("Content")
        or v.DESCRIPTOR.name.endswith("Options")
    )
)

_CONTENT_DESCRIPTOR_NAMES: frozenset[str] = frozenset(
    cls.DESCRIPTOR.name for cls in _CONTENT_PROTO_TYPES
)

_MEDIA_TYPE_MAP = {
    content_pb2.VideoContent: "video",
    content_pb2.AudioContent: "audio",
    content_pb2.ImageContent: "image",
    content_pb2.DocumentContent: "document",
    content_pb2.TextContent: "text",
}

_MEDIA_DESCRIPTOR_MAP = {
    "VideoContent": "video",
    "AudioContent": "audio",
    "ImageContent": "image",
    "DocumentContent": "document",
    "TextContent": "text",
}


def _is_content_proto(obj: Any) -> bool:
  """Returns True if obj is an interactions Content proto or media component."""
  if isinstance(obj, _CONTENT_PROTO_TYPES):
    return True
  if (
      hasattr(obj, "DESCRIPTOR")
      and obj.DESCRIPTOR.name in _CONTENT_DESCRIPTOR_NAMES
  ):
    return True
  return False


def _as_mapping(obj: Any) -> abc.Mapping[str, Any] | None:
  """Converts mappings, Pydantic models, and dataclass instances to a Mapping, or None if not structured."""
  if _is_content_proto(obj) or isinstance(obj, type):
    return None
  if isinstance(obj, abc.Mapping):
    return obj
  if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")):
    try:
      d = obj.model_dump()
      if isinstance(d, abc.Mapping):
        return d
    except (TypeError, ValueError, AttributeError):
      pass
  if hasattr(obj, "dict") and callable(getattr(obj, "dict")):
    try:
      d = obj.dict()
      if isinstance(d, abc.Mapping):
        return d
    except (TypeError, ValueError, AttributeError):
      pass
  if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
    try:
      return {f.name: getattr(obj, f.name) for f in dataclasses.fields(obj)}
    except (TypeError, ValueError, AttributeError):
      pass
  return None


def is_structured(obj: Any) -> bool:
  """Returns True if obj is a mapping, Pydantic model, or dataclass instance."""
  return _as_mapping(obj) is not None


def has_proto_extensions(obj: Any) -> bool:
  """Checks whether an object or its descendants contain genai Content components.

  Args:
    obj: The Python object or data structure to inspect.

  Returns:
    True if the object or any nested value is a Content or media Content proto,
    otherwise False.
  """
  if _is_content_proto(obj):
    return True
  m = _as_mapping(obj)
  if m is not None:
    return any(has_proto_extensions(v) for v in m.values())
  if isinstance(obj, abc.Sequence) and not isinstance(obj, (str, bytes)):
    return any(has_proto_extensions(v) for v in obj)
  return False


def to_struct(obj: Any) -> content_pb2.Struct:
  """Converts a mapping, Pydantic model, or dataclass into a content_pb2.Struct protobuf message.

  Args:
    obj: The Python mapping, Pydantic model, or dataclass to serialize.

  Returns:
    A populated content_pb2.Struct protobuf message.
  """
  if isinstance(obj, content_pb2.Struct) or (
      hasattr(obj, "DESCRIPTOR") and obj.DESCRIPTOR.name == "Struct"
  ):
    return obj

  m = _as_mapping(obj)
  if m is not None:
    fields = [
        content_pb2.Field(name=str(k), value=value_to_proto(v))
        for k, v in m.items()
    ]
    return content_pb2.Struct(fields=fields)

  return content_pb2.Struct(
      fields=[content_pb2.Field(name="result", value=value_to_proto(obj))]
  )


def dict_to_struct(d: abc.Mapping[str, Any]) -> content_pb2.Struct:
  """Converts a Python mapping into a content_pb2.Struct protobuf message.

  Args:
    d: The Python mapping to serialize.

  Returns:
    A populated content_pb2.Struct protobuf message.
  """
  return to_struct(d)


def _wrap_media_content(field_name: str, val: Any) -> content_pb2.Content:
  """Constructs a Content message with the given field set."""
  content_cls = cast(Any, content_pb2.Content)
  return content_cls(**{field_name: val})


def value_to_proto(val: Any) -> content_pb2.Value:
  """Converts a Python value into a content_pb2.Value protobuf message.

  Args:
    val: The Python value or data structure to serialize.

  Returns:
    A populated content_pb2.Value protobuf message.
  """
  if val is None:
    return content_pb2.Value(null_value="NULL_VALUE")
  if isinstance(val, bool):
    return content_pb2.Value(bool_value=val)
  if isinstance(val, (int, float)):
    return content_pb2.Value(number_value=float(val))
  if isinstance(val, str):
    return content_pb2.Value(string_value=val)
  if isinstance(val, abc.Sequence) and not isinstance(val, (str, bytes)):
    return content_pb2.Value(
        list_value=content_pb2.ListValue(
            values=[value_to_proto(x) for x in val]
        )
    )

  m = _as_mapping(val)
  if m is not None:
    return content_pb2.Value(struct_value=to_struct(m))

  if isinstance(val, content_pb2.Value):
    return val
  if isinstance(val, content_pb2.Struct):
    return content_pb2.Value(struct_value=val)
  if isinstance(val, content_pb2.ListValue):
    return content_pb2.Value(list_value=val)
  if isinstance(val, content_pb2.Content):
    return content_pb2.Value(content_value=val)

  for media_type, field_name in _MEDIA_TYPE_MAP.items():
    if isinstance(val, media_type):
      return content_pb2.Value(
          content_value=_wrap_media_content(field_name, val)
      )

  if hasattr(val, "DESCRIPTOR"):
    val_any = cast(Any, val)
    desc_name = val_any.DESCRIPTOR.name
    if desc_name == "Value":
      return val_any
    elif desc_name == "Struct":
      return content_pb2.Value(struct_value=val_any)
    elif desc_name == "ListValue":
      return content_pb2.Value(list_value=val_any)
    elif desc_name == "Content":
      return content_pb2.Value(content_value=val_any)
    elif desc_name in _MEDIA_DESCRIPTOR_MAP:
      return content_pb2.Value(
          content_value=_wrap_media_content(
              _MEDIA_DESCRIPTOR_MAP[desc_name], val_any
          )
      )

  return content_pb2.Value(string_value=str(val))


def _get_display_name(obj: Any) -> str:
  """Extracts display_name from a Content or media component object if available."""
  try:
    if hasattr(obj, "display_name") and obj.display_name:
      return obj.display_name
  except (AttributeError, ValueError):
    pass
  if hasattr(obj, "WhichOneof"):
    try:
      oneof_field = obj.WhichOneof("type")
      if oneof_field:
        inner = getattr(obj, oneof_field)
        if hasattr(inner, "display_name") and inner.display_name:
          return inner.display_name
    except (AttributeError, ValueError):
      pass
  return ""


def _get_uri(obj: Any) -> str:
  """Extracts uri from a Content or media component object if available."""
  try:
    if hasattr(obj, "uri") and obj.uri:
      return obj.uri
  except (AttributeError, ValueError):
    pass
  if hasattr(obj, "WhichOneof"):
    try:
      oneof_field = obj.WhichOneof("type")
      if oneof_field:
        inner = getattr(obj, oneof_field)
        if hasattr(inner, "uri") and inner.uri:
          return inner.uri
    except (AttributeError, ValueError):
      pass
  return ""


def to_json_fallback(obj: Any) -> Any:
  """Converts objects into JSON-serializable primitives for fallback.

  Args:
    obj: The Python object or data structure to convert.

  Returns:
    A JSON-serializable representation, replacing Content instances with
    reference pointers if display_name or uri is present.
  """
  if _is_content_proto(obj):
    dn = _get_display_name(obj)
    if dn:
      return {"$ref": dn}
    uri = _get_uri(obj)
    if uri:
      return {"$ref": uri}
    return str(obj)

  if isinstance(obj, content_pb2.Struct) or (
      hasattr(obj, "DESCRIPTOR") and obj.DESCRIPTOR.name == "Struct"
  ):
    return {f.name: to_json_fallback(f.value) for f in obj.fields}

  if isinstance(obj, content_pb2.ListValue) or (
      hasattr(obj, "DESCRIPTOR") and obj.DESCRIPTOR.name == "ListValue"
  ):
    return [to_json_fallback(v) for v in obj.values]

  if isinstance(obj, content_pb2.Value) or (
      hasattr(obj, "DESCRIPTOR") and obj.DESCRIPTOR.name == "Value"
  ):
    which = obj.WhichOneof("kind")
    if which == "null_value":
      return None
    elif which == "bool_value":
      return obj.bool_value
    elif which == "number_value":
      return obj.number_value
    elif which == "string_value":
      return obj.string_value
    elif which == "struct_value":
      return to_json_fallback(obj.struct_value)
    elif which == "list_value":
      return to_json_fallback(obj.list_value)
    elif which == "content_value":
      return to_json_fallback(obj.content_value)
    return str(obj)

  m = _as_mapping(obj)
  if m is not None:
    return {k: to_json_fallback(v) for k, v in m.items()}

  if isinstance(obj, abc.Sequence) and not isinstance(obj, (str, bytes)):
    return [to_json_fallback(v) for v in obj]

  if isinstance(obj, (str, int, float, bool, type(None))):
    return obj

  return str(obj)
